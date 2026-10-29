"""A schema-stale cached author result is a cache miss, never a campaign crash.

Rung-12 field failure: the wave-8 scalar-seam tightening of the author-proposal
schema (embedded in ``author-result.v4``) made the LIVE result schema refuse
HISTORICAL author results carrying the old ``constructed``/``categorical``
spelling. The canonical-artifact-authority rehydration path validates cached
results against the live schema inside ``validate_artifact_checkpoint``, so ONE
stale transaction aborted the whole 100-model campaign with
``DriverIntegrationError: canonical artifact authority cannot be rehydrated``.

The contract pinned here: the solve cache / artifact authority is
reuse-convenience, not history. The live schema governs REUSE eligibility, and
ineligibility is a per-transaction stale entry -- quarantined out of the
projection's reusable ``transactions``, resolved as ``None`` (fresh
authorship), and disclosed with a typed operational event -- while every byte
anchor keeps verifying and every OTHER model's authority stays alive. Ledger
records themselves keep validating at their recorded basis: the historical
model/gate schemas deliberately kept the open input-leaf shape.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

import menagerie.crawler.artifact_transactions as artifact_transactions_module
import menagerie.crawler.tests.test_artifact_transactions as artifact_fixtures
from menagerie.crawler.artifact_transactions import (
    ArtifactCheckpointProjection,
    ArtifactRehydrationError,
    ArtifactTransactionId,
    StaleArtifactTransaction,
    resolve_final_artifact_transaction,
    validate_artifact_checkpoint,
)
from menagerie.crawler.constants import (
    MODEL_SCHEMA_VERSION_V3,
    OPERATIONAL_EVENT_SCHEMA_VERSION,
)
from menagerie.crawler.driver_admission import (
    ARTIFACT_AUTHORITY_SCHEMA_STALE_DISPOSITION,
)
from menagerie.crawler.identity import stable_hash
from menagerie.crawler.mirrors import MirrorStore
from menagerie.crawler.recordio import JsonlLedger, scan_jsonl
from menagerie.crawler.schema import validate_payload
from menagerie.crawler.tests.conftest import make_model
from menagerie.crawler.tests.test_artifact_transactions import (
    _commit_private,
    _context,
    _mirrors,
)
from menagerie.crawler.tests.test_slice_f_driver import _driver, _paths, _snapshot

# The exact historical input-leaf spelling from the rung-12 crash (the
# superseded m9304/m9407/m9653/m9666/m9678-era smp proposals).
_HISTORICAL_LITERAL_KWARG_LEAF: dict[str, Any] = {
    "path": "kwargs.encoder_name",
    "kind": "constructed",
    "semantic_role": "constructor kwarg selecting the se_resnet50 encoder architecture",
    "shape": [],
    "dtype": "not-applicable",
    "device_policy": "not-applicable",
    "distribution": "categorical",
    "constraints": ['fixed value "se_resnet50" selecting the SENetEncoder registry entry'],
    "source_evidence_ids": ["ev-pan-signature"],
}


_REAL_BLOCKED_RESULT = artifact_fixtures._blocked_result


def _historically_valid_result(
    stable_id: str, context: Any, source_manifest: dict[str, Any]
) -> dict[str, Any]:
    """Build a result the OLD schema accepted and the LIVE schema refuses.

    History simulation: the artifact machinery wrote these bytes under the
    pre-tightening schema; the mutation below is refused by the CURRENT
    ``author-result.v4`` validator exactly like the field specimens' embedded
    ``constructed``/``categorical`` proposals.

    Parameters
    ----------
    stable_id, context, source_manifest:
        Exact model, active trust roots, and source manifest.

    Returns
    -------
    dict[str, Any]
        Self-hashed result mapping refused by the live result schema.
    """

    result = _REAL_BLOCKED_RESULT(stable_id, context, source_manifest)
    result["payload"] = dict(result["payload"])
    result["payload"]["input_contract_spelling"] = deepcopy(_HISTORICAL_LITERAL_KWARG_LEAF)
    result["result_sha256"] = stable_hash(
        {key: value for key, value in result.items() if key != "result_sha256"}
    )
    return result


def _campaign_roots(tmp_path: Path) -> tuple[Any, MirrorStore, Any, Path, Path]:
    """Build the shared write-side fixtures for two-transaction campaigns.

    Parameters
    ----------
    tmp_path:
        Per-test scratch root.

    Returns
    -------
    tuple
        Authority context, mirrors, artifact ledger, canonical root, and
        repository root.
    """

    context = _context("m_clean", "m_stale")
    mirrors = _mirrors(tmp_path)
    canonical_root = tmp_path / "repo" / "menagerie" / "crawler"
    canonical_root.mkdir(parents=True, exist_ok=True)
    ledger = artifact_transactions_module.ArtifactEventLedger(
        tmp_path / "artifacts" / "events.jsonl"
    )
    return context, mirrors, ledger, canonical_root, tmp_path / "repo"


def _write_clean_and_stale_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Any, MirrorStore, Path, Path]:
    """Write one clean and one pre-tightening transaction with real machinery.

    The stale transaction's bytes are written EXACTLY as history wrote them:
    through the real stage/authorize/publish writers, with the live schema
    check suspended for the write only (the old schema accepted these bytes).

    Parameters
    ----------
    tmp_path, monkeypatch:
        Scratch root and an active monkeypatch context.

    Returns
    -------
    tuple
        Ledger, mirrors, canonical root, and repository root for read passes.
    """

    context, mirrors, ledger, canonical_root, repository_root = _campaign_roots(tmp_path)
    _commit_private(
        "m_clean",
        content=b"clean upstream bytes",
        context=context,
        mirrors=mirrors,
        ledger=ledger,
        canonical_root=canonical_root,
        repository_root=repository_root,
        evidence_id="ev-source",
    )
    with pytest.MonkeyPatch.context() as write_era:
        write_era.setattr(artifact_fixtures, "_blocked_result", _historically_valid_result)
        write_era.setattr(
            artifact_transactions_module, "validate_payload", lambda *args, **kwargs: None
        )
        _commit_private(
            "m_stale",
            content=b"stale upstream bytes",
            context=context,
            mirrors=mirrors,
            ledger=ledger,
            canonical_root=canonical_root,
            repository_root=repository_root,
            evidence_id="ev-source",
        )
    return ledger, mirrors, canonical_root, repository_root


def test_schema_stale_transaction_quarantines_without_failing_the_projection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One pre-tightening transaction never aborts the campaign projection.

    The rung-12 regression: the projection must survive, quarantine exactly
    the stale entry with the live validator's own violation text, and keep
    the clean sibling fully reusable.
    """

    ledger, mirrors, canonical_root, repository_root = _write_clean_and_stale_history(
        tmp_path, monkeypatch
    )

    projection = validate_artifact_checkpoint(
        (ledger.path,),
        context=_context("m_clean", "m_stale"),
        mirrors=mirrors,
        canonical_root=canonical_root,
        repository_root=repository_root,
    )

    clean_keys = [key for key in projection.transactions if key[0] == "m_clean"]
    stale_keys = [key for key in projection.stale_transactions if key[0] == "m_stale"]
    assert len(clean_keys) == 1
    assert len(stale_keys) == 1
    assert not [key for key in projection.transactions if key[0] == "m_stale"]
    stale = projection.stale_transactions[stale_keys[0]]
    assert "invalid typed author result" in stale.reason
    # The quarantined transaction's physical custody still participates: the
    # checkpoint's own mirror-inventory pass ran to completion above, and the
    # stale entry retains its byte-verified recorded claims for record-anchored
    # consumers.
    assert stale.claims

    # (b) the valid cached authority still resolves for reuse.
    clean_key = clean_keys[0]
    resolved = resolve_final_artifact_transaction(
        projection,
        stable_id="m_clean",
        work_id=clean_key[1],
        transaction_id=clean_key[2],
    )
    assert resolved is not None and resolved.stable_id == "m_clean"
    # The transaction carries its own record-time prompt pins, so rehydration
    # consumers can validate history at the recorded basis instead of
    # re-imposing the live prompt files (rung-12 leg 2).
    context = _context("m_clean", "m_stale")
    assert resolved.recorded_checker_prompt_identity == context.checker_prompt_identity
    assert resolved.recorded_author_prompt_identity == context.author_prompt_identity

    # (a) the stale recorded authority reads as a cache miss -- for the exact
    # recorded transaction id and for unrecorded selection alike -- so the
    # caller routes the model to fresh authorship instead of crashing.
    stale_key = stale_keys[0]
    assert (
        resolve_final_artifact_transaction(
            projection,
            stable_id="m_stale",
            work_id=stale_key[1],
            transaction_id=stale_key[2],
        )
        is None
    )
    assert (
        resolve_final_artifact_transaction(projection, stable_id="m_stale", work_id=stale_key[1])
        is None
    )
    # A genuinely absent recorded transaction is still an error, not a miss:
    # the stale arm never widens into permission to ignore recorded authority.
    with pytest.raises(ArtifactRehydrationError):
        resolve_final_artifact_transaction(
            projection,
            stable_id="m_stale",
            work_id=stale_key[1],
            transaction_id=ArtifactTransactionId("txn-" + "0" * 24),
        )


def test_vector_contradicting_its_record_quarantines_that_transaction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The rung-12 leg-4 poison: a vector/gate contradiction is one model's defect.

    m1359's final event recorded the metadata gate while its document vector
    pinned the fidelity gate; the projection consult aborted the whole
    campaign. The contradiction must still refuse -- as a quarantined stale
    entry (no reuse, loud) -- while the projection survives for every other
    model.
    """

    context = _context("m_clean", "m_poisoned")
    mirrors = _mirrors(tmp_path)
    canonical_root = tmp_path / "repo" / "menagerie" / "crawler"
    canonical_root.mkdir(parents=True, exist_ok=True)
    ledger = artifact_transactions_module.ArtifactEventLedger(
        tmp_path / "artifacts" / "events.jsonl"
    )
    _commit_private(
        "m_clean",
        content=b"clean upstream bytes",
        context=context,
        mirrors=mirrors,
        ledger=ledger,
        canonical_root=canonical_root,
        repository_root=tmp_path / "repo",
        evidence_id="ev-source",
    )
    real_vector = artifact_fixtures._vector

    def poisoned_vector(*args: Any, **kwargs: Any) -> Any:
        """Pin a checker gate the final event does not record."""

        kwargs["checker_gate_identity"] = "gate-fidelity-generation-3"
        return real_vector(*args, **kwargs)

    with pytest.MonkeyPatch.context() as write_era:
        write_era.setattr(artifact_fixtures, "_vector", poisoned_vector)
        _commit_private(
            "m_poisoned",
            content=b"poisoned upstream bytes",
            context=context,
            mirrors=mirrors,
            ledger=ledger,
            canonical_root=canonical_root,
            repository_root=tmp_path / "repo",
            evidence_id="ev-source",
        )

    projection = validate_artifact_checkpoint(
        (ledger.path,),
        context=context,
        mirrors=mirrors,
        canonical_root=canonical_root,
        repository_root=tmp_path / "repo",
    )

    assert [key[0] for key in projection.transactions] == ["m_clean"]
    poisoned = [
        stale for key, stale in projection.stale_transactions.items() if key[0] == "m_poisoned"
    ]
    assert len(poisoned) == 1
    assert "contradicts its record at checker_gate_identity" in poisoned[0].reason
    assert poisoned[0].claims
    poisoned_key = next(key for key in projection.stale_transactions if key[0] == "m_poisoned")
    assert (
        resolve_final_artifact_transaction(
            projection,
            stable_id="m_poisoned",
            work_id=poisoned_key[1],
            transaction_id=poisoned_key[2],
        )
        is None
    )


def test_interrupted_finalization_quarantines_instead_of_failing_the_projection(
    tmp_path: Path,
) -> None:
    """A crash between authorization and commitment never bricks later legs.

    The chain stops at two events (staged, authorized). Its private custody
    stays registered exactly like durable pending custody, the transaction
    quarantines as incomplete, and the projection completes.
    """

    context = _context("m_clean", "m_interrupted")
    mirrors = _mirrors(tmp_path)
    canonical_root = tmp_path / "repo" / "menagerie" / "crawler"
    canonical_root.mkdir(parents=True, exist_ok=True)
    ledger = artifact_transactions_module.ArtifactEventLedger(
        tmp_path / "artifacts" / "events.jsonl"
    )
    _commit_private(
        "m_clean",
        content=b"clean upstream bytes",
        context=context,
        mirrors=mirrors,
        ledger=ledger,
        canonical_root=canonical_root,
        repository_root=tmp_path / "repo",
        evidence_id="ev-source",
    )
    real_publish = artifact_transactions_module.publish_authorized_artifact

    def interrupted_publish(*args: Any, **kwargs: Any) -> Any:
        """Simulate a crash after authorization, before commitment."""

        raise RuntimeError("synthetic crash before commitment")

    with pytest.MonkeyPatch.context() as crash_era:
        crash_era.setattr(artifact_fixtures, "publish_authorized_artifact", interrupted_publish)
        with pytest.raises(RuntimeError, match="synthetic crash"):
            _commit_private(
                "m_interrupted",
                content=b"interrupted upstream bytes",
                context=context,
                mirrors=mirrors,
                ledger=ledger,
                canonical_root=canonical_root,
                repository_root=tmp_path / "repo",
                evidence_id="ev-source",
            )
    assert real_publish is artifact_transactions_module.publish_authorized_artifact

    projection = validate_artifact_checkpoint(
        (ledger.path,),
        context=context,
        mirrors=mirrors,
        canonical_root=canonical_root,
        repository_root=tmp_path / "repo",
    )

    assert [key[0] for key in projection.transactions] == ["m_clean"]
    interrupted = [
        stale for key, stale in projection.stale_transactions.items() if key[0] == "m_interrupted"
    ]
    assert len(interrupted) == 1
    # AT-7 refined the disclosed reason for THIS shape (staged + committed
    # authorization, no final event): it is the RESUMABLE torn chain --
    # resume_torn_publication's exact input. The quarantine itself is
    # unchanged: never projected authority, claims empty, no final resolve.
    assert "torn-chain-resumable" in interrupted[0].reason
    assert interrupted[0].claims == ()
    assert (
        resolve_final_artifact_transaction(
            projection, stable_id="m_interrupted", work_id="work-m_interrupted"
        )
        is None
    )


def _minimal_stale_entry(
    stable_id: str, work_id: str, transaction_id: str
) -> StaleArtifactTransaction:
    """Build one minimal stale entry for driver-seam disclosure tests.

    Parameters
    ----------
    stable_id, work_id, transaction_id:
        Exact projection key components the entry is filed under.

    Returns
    -------
    StaleArtifactTransaction
        Entry carrying a recognizable live-schema violation summary.
    """

    return StaleArtifactTransaction(
        stable_id=stable_id,
        work_id=work_id,
        transaction_id=ArtifactTransactionId(transaction_id),
        claims=(),
        reason=(
            "invalid typed author result: menagerie.crawler.author-result.v4 "
            "validation failed: 'categorical' is not one of ['constructor']"
        ),
    )


def test_stale_authority_disclosure_is_typed_loud_and_once_per_run(tmp_path: Path) -> None:
    """The cache death writes one schema-valid operational event per stale entry."""

    snapshot = _snapshot(tmp_path, count=1)
    paths = _paths(tmp_path, snapshot)
    driver = _driver(tmp_path, snapshot)
    item = driver._ordered_work(snapshot, {})[0]
    transaction_id = "txn-" + "b" * 24
    projection = ArtifactCheckpointProjection(
        transactions={},
        objects=(),
        claims=(),
        stale_transactions={
            (item.stable_id, item.active_work_id, ArtifactTransactionId(transaction_id)): (
                _minimal_stale_entry(item.stable_id, item.active_work_id, transaction_id)
            ),
            ("m_other_model", "work-other", ArtifactTransactionId("txn-" + "c" * 24)): (
                _minimal_stale_entry("m_other_model", "work-other", "txn-" + "c" * 24)
            ),
        },
    )
    operational = JsonlLedger(paths.operational_ledger, OPERATIONAL_EVENT_SCHEMA_VERSION)

    driver._report_stale_artifact_authority(item, projection, operational)
    driver._report_stale_artifact_authority(item, projection, operational)

    events = scan_jsonl(paths.operational_ledger)
    stale_events = [
        event
        for event in events
        if event.get("details", {}).get("disposition")
        == ARTIFACT_AUTHORITY_SCHEMA_STALE_DISPOSITION
    ]
    # Once per run per entry, scoped to THIS item's model only.
    assert len(stale_events) == 1
    details = stale_events[0]["details"]
    assert details["stable_id"] == item.stable_id
    assert details["work_id"] == item.active_work_id
    assert details["transaction_id"] == transaction_id
    assert "'categorical' is not one of ['constructor']" in details["violation_summary"]
    assert "fresh authorship" in details["remedy"]
    assert stale_events[0]["event_id"].startswith("campaign-health-")


def test_historical_model_rows_still_validate_on_ledger_read() -> None:
    """(c) Model records keep the open input-leaf shape: history loads unchanged.

    The reuse-eligibility tightening is scoped to the authoring surface; a
    historical model row carrying the old ``constructed``/``categorical``
    spelling (the exact superseded m9666-era shape) must keep validating
    against the live model schema exactly as before, so ledger reads of the
    pilot campaign never turn into ``LedgerCorruptionError``.
    """

    row = make_model("m9666", accepted=True)
    contract = row["input_contract"]
    contract["kwargs"] = [deepcopy(_HISTORICAL_LITERAL_KWARG_LEAF)]
    validate_payload(row, MODEL_SCHEMA_VERSION_V3)
