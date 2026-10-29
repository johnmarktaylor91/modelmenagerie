"""W5 checkpoint cache: the final-authority derivation stops being O(history).

risk_r5 root cause (measured on the pilot corpus): every
``_rehydrate_final_authority`` call re-ran ``validate_artifact_checkpoint``
over the COMPLETE artifact history -- ~7s per model at 653 events, 88% of it
jsonschema re-validation of unchanged, content-addressed records -- so the
per-model close cost grew with terminal count and the campaign close was
O(N^2).

The contract pinned here:

- A persistent ``ValidationMemo`` (derived, deletable, outside the records
  ledgers) memoizes ONLY positive outcomes of pure-function validations,
  keyed by live-reverified content identity plus the schema/code environment
  identity. It can never excuse content-identity verification, never change
  a verdict, and degrades to live derivation on corruption or staleness.
- ``ArtifactCheckpointIncrement`` advances a fully validated projection by
  folding only delta events; the result is equal to full validation of the
  same history, per-advance validation work is O(delta) not O(history), and
  global-registry refusals survive the incremental path.
- Cold start (no cache anywhere) reproduces today's behavior exactly.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Optional

import pytest

import menagerie.crawler.artifact_transactions as artifact_transactions_module
import menagerie.crawler.recordio as recordio_module
import menagerie.crawler.tests.test_artifact_transactions as artifact_fixtures
from menagerie.crawler.artifact_transactions import (
    ArtifactCheckpointIncrement,
    ArtifactEventKind,
    ArtifactEventLedger,
    ArtifactInput,
    ArtifactTransitionError,
    ReconstructionInputs,
    append_artifact_authorization,
    derive_artifact_claims,
    derive_publication_authorization_id,
    publish_authorized_artifact,
    stage_private_artifact,
    staged_artifact_for_result,
    validate_artifact_checkpoint,
)
from menagerie.crawler.authority import PublicationAuthorization
from menagerie.crawler.checkpoint_cache import (
    CHECKPOINT_CACHE_EPOCH,
    ValidationMemo,
    checkpoint_cache_environment_identity,
)
from menagerie.crawler.identity import hash_bytes, stable_hash
from menagerie.crawler.licenses import LicenseDecision, RedistributionClass
from menagerie.crawler.mirrors import ArtifactOrigin
from menagerie.crawler.recordio import LedgerCorruptionError
from menagerie.crawler.tests.test_artifact_authority_schema_stale import (
    _historically_valid_result,
)
from menagerie.crawler.tests.test_artifact_transactions import (
    NOW,
    _blocked_result,
    _commit_private,
    _context,
    _gate_item,
    _mirrors,
    _source_manifest,
    _vector,
)

MODELS = tuple(f"m_zoo{index}" for index in range(12)) + ("m_stale",)


def _campaign(tmp_path: Path) -> dict[str, Any]:
    """Build one empty campaign world with real mirrors and ledger.

    Parameters
    ----------
    tmp_path:
        Per-test scratch root.

    Returns
    -------
    dict[str, Any]
        Context, mirrors, ledger, canonical root, and repository root.
    """

    canonical_root = tmp_path / "repo" / "menagerie" / "crawler"
    canonical_root.mkdir(parents=True, exist_ok=True)
    return {
        "context": _context(*MODELS),
        "mirrors": _mirrors(tmp_path),
        "ledger": ArtifactEventLedger(canonical_root / "records" / "artifacts" / "events.jsonl"),
        "canonical_root": canonical_root,
        "repository_root": tmp_path / "repo",
    }


def _checkpoint_kwargs(campaign: dict[str, Any], memo: Optional[ValidationMemo] = None) -> dict:
    """Return the full-validation keyword bundle for one campaign world.

    Parameters
    ----------
    campaign:
        World built by :func:`_campaign`.
    memo:
        Optional persistent validation memo.

    Returns
    -------
    dict
        Keyword arguments for :func:`validate_artifact_checkpoint`.
    """

    kwargs = {
        "context": campaign["context"],
        "mirrors": campaign["mirrors"],
        "canonical_root": campaign["canonical_root"],
        "repository_root": campaign["repository_root"],
    }
    if memo is not None:
        kwargs["memo"] = memo
    return kwargs


def _full(campaign: dict[str, Any], memo: Optional[ValidationMemo] = None):
    """Run one full checkpoint validation from the on-disk shard.

    Parameters
    ----------
    campaign, memo:
        Campaign world and optional memo.

    Returns
    -------
    ArtifactCheckpointProjection
        Fully validated projection.
    """

    return validate_artifact_checkpoint(
        (campaign["ledger"].path,), **_checkpoint_kwargs(campaign, memo)
    )


def _commit(campaign: dict[str, Any], stable_id: str, content: bytes) -> None:
    """Fully commit one private transaction with the real machinery.

    Parameters
    ----------
    campaign, stable_id, content:
        Campaign world, model identity, and exact private bytes.
    """

    _commit_private(
        stable_id,
        content=content,
        context=campaign["context"],
        mirrors=campaign["mirrors"],
        ledger=campaign["ledger"],
        canonical_root=campaign["canonical_root"],
        repository_root=campaign["repository_root"],
        evidence_id="ev-source",
    )


def _stage_only(campaign: dict[str, Any], stable_id: str, content: bytes) -> dict[str, Any]:
    """Stage private custody without authorizing or committing.

    Parameters
    ----------
    campaign, stable_id, content:
        Campaign world, model identity, and exact private bytes.

    Returns
    -------
    dict[str, Any]
        Staged artifact plus the exact inputs later transitions replay.
    """

    source_manifest = _source_manifest(content)
    result = _blocked_result(stable_id, campaign["context"], source_manifest)
    digest = hash_bytes(content)
    staged = stage_private_artifact(
        (
            ArtifactInput(
                content=content,
                content_sha256=digest,
                logical_role="source",
                logical_path=f"menagerie/crawler/source_cas/{digest[7:]}.source",
                source_id="src-upstream",
                origin=ArtifactOrigin("https://example.test/source.tar", "v1"),
                fetch_recipe="GET exact source.tar@v1",
                evidence_ids=("ev-source",),
                media_type="application/x-tar",
            ),
        ),
        context=campaign["context"],
        stable_id=stable_id,
        work_id=f"work-{stable_id}",
        author_result=result,
        proposal=None,
        source_manifest=source_manifest,
        mirrors=campaign["mirrors"],
        ledger=campaign["ledger"],
        created_at=NOW,
    )
    return {
        "staged": staged,
        "result": result,
        "source_manifest": source_manifest,
        "content_sha256": digest,
    }


def _authorize(campaign: dict[str, Any], pending: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    """Append terminal authorization for a previously staged transaction.

    Parameters
    ----------
    campaign, pending:
        Campaign world and the staged transaction's exact inputs.

    Returns
    -------
    tuple[PublicationAuthorization, dict[str, Any]]
        The appended authorization and the accepted gate item.
    """

    context = campaign["context"]
    staged = pending["staged"]
    result = pending["result"]
    source_manifest = pending["source_manifest"]
    stable_id = staged.custody_claims[0].stable_id
    gate_item = _gate_item(stable_id, result, source_manifest["manifest_sha256"])
    provisional_vector = _vector(
        context,
        stable_id,
        str(staged.transaction_id),
        result["result_id"],
        source_manifest["manifest_sha256"],
    )
    decision = LicenseDecision(
        content_sha256=pending["content_sha256"],
        redistribution_class=RedistributionClass.RESTRICTED_PRIVATE,
        evidence_ids=("ev-source",),
        rationale="restricted fixture",
    )
    decisions = {staged.custody_claims[0].claim_id: decision}
    authorization_id = derive_publication_authorization_id(
        staged,
        accepted_gate_id="gate-terminal",
        accepted_gate_item_sha256=stable_hash(gate_item),
        dependency_vector=provisional_vector,
        decisions=decisions,
        publication_policy_identity=context.publication_policy_identity,
    )
    claims = derive_artifact_claims(
        staged,
        accepted_gate_id="gate-terminal",
        authorization_id=authorization_id,
        decisions=decisions,
        mirrors=campaign["mirrors"],
    )
    vector = replace(
        provisional_vector, artifact_claim_ids=tuple(claim.claim_id for claim in claims)
    )
    authorization = PublicationAuthorization(
        authorization_id=authorization_id,
        stable_id=stable_id,
        work_id=f"work-{stable_id}",
        transaction_id=staged.transaction_id,
        accepted_gate_id="gate-terminal",
        accepted_gate_item_sha256=stable_hash(gate_item),
        dependency_vector=vector,
        claim_ids=tuple(claim.claim_id for claim in claims),
        public_object_ids=(),
        private_object_ids=tuple(claim.object_id for claim in claims),
        publication_policy_identity=context.publication_policy_identity,
    )
    append_artifact_authorization(
        staged,
        authorization,
        claims,
        accepted_gate_item=gate_item,
        event_kind=ArtifactEventKind.TERMINAL_AUTHORIZED,
        context=context,
        mirrors=campaign["mirrors"],
        ledger=campaign["ledger"],
        created_at=NOW,
    )
    return authorization, gate_item


def _finalize(
    campaign: dict[str, Any],
    pending: dict[str, Any],
    authorization: Any,
    gate_item: dict[str, Any],
) -> None:
    """Privately commit a previously authorized transaction.

    Parameters
    ----------
    campaign, pending, authorization, gate_item:
        Campaign world and the transaction's exact recorded inputs.
    """

    publish_authorized_artifact(
        pending["staged"],
        authorization,
        reconstruction_inputs=ReconstructionInputs(
            author_result=pending["result"],
            proposal=None,
            handoff_execution=None,
            source_manifest=pending["source_manifest"],
            accepted_gate_item=gate_item,
        ),
        context=campaign["context"],
        mirrors=campaign["mirrors"],
        ledger=campaign["ledger"],
        canonical_root=campaign["canonical_root"],
        repository_root=campaign["repository_root"],
        created_at=NOW,
    )


def _commit_schema_stale(campaign: dict[str, Any], stable_id: str, content: bytes) -> None:
    """Commit one pre-tightening transaction exactly as history wrote it.

    Parameters
    ----------
    campaign, stable_id, content:
        Campaign world, model identity, and exact private bytes.
    """

    with pytest.MonkeyPatch.context() as write_era:
        write_era.setattr(artifact_fixtures, "_blocked_result", _historically_valid_result)
        write_era.setattr(
            artifact_transactions_module, "validate_payload", lambda *args, **kwargs: None
        )
        _commit(campaign, stable_id, content)


class _SchemaValidationCounter:
    """Count live jsonschema validations across every import site."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Wrap ``validate_payload`` in every consuming module.

        Parameters
        ----------
        monkeypatch:
            Active per-test patch context.
        """

        self.calls = 0
        real = artifact_transactions_module.validate_payload

        def counting(payload: Any, schema_version: Any = None) -> None:
            self.calls += 1
            return real(payload, schema_version)

        monkeypatch.setattr(artifact_transactions_module, "validate_payload", counting)
        monkeypatch.setattr(recordio_module, "validate_payload", counting)

    def take(self) -> int:
        """Return calls since the previous take.

        Returns
        -------
        int
            Live schema validations observed.
        """

        count = self.calls
        self.calls = 0
        return count


# ---------------------------------------------------------------------------
# ValidationMemo: persistence, environment binding, degrade-to-live
# ---------------------------------------------------------------------------


def test_validation_memo_round_trip_and_cold_start(tmp_path: Path) -> None:
    """A cold memo is empty; recorded keys survive flush and reopen."""

    root = tmp_path / "derived" / "checkpoint-cache"
    memo = ValidationMemo.open(root)
    assert not memo.seen("record", "sha256:" + "1" * 64)
    assert memo.diagnostics == []

    memo.record("record", "sha256:" + "1" * 64)
    memo.record("transaction-document", "sha256:" + "2" * 64)
    assert memo.seen("record", "sha256:" + "1" * 64)
    # A key is never visible under a foreign kind.
    assert not memo.seen("transaction-document", "sha256:" + "1" * 64)
    memo.flush()

    reopened = ValidationMemo.open(root)
    assert reopened.seen("record", "sha256:" + "1" * 64)
    assert reopened.seen("transaction-document", "sha256:" + "2" * 64)
    assert reopened.diagnostics == []


def test_validation_memo_ignores_foreign_environment_entries(tmp_path: Path) -> None:
    """Entries recorded under another schema/code environment never hit."""

    root = tmp_path / "derived" / "checkpoint-cache"
    memo = ValidationMemo.open(root)
    memo.record("record", "sha256:" + "3" * 64)
    memo.flush()

    path = memo.path
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert rows and all(
        row["environment"] == checkpoint_cache_environment_identity() for row in rows
    )
    stale = dict(rows[0])
    stale["environment"] = "sha256:" + "f" * 64
    stale["key"] = "sha256:" + "4" * 64
    with path.open("a") as handle:
        handle.write(json.dumps(stale) + "\n")

    reopened = ValidationMemo.open(root)
    assert reopened.seen("record", "sha256:" + "3" * 64)
    assert not reopened.seen("record", "sha256:" + "4" * 64)


def test_validation_memo_corrupt_file_degrades_to_empty_with_diagnostic(tmp_path: Path) -> None:
    """A corrupt complete line discards the memo loudly and never raises."""

    root = tmp_path / "derived" / "checkpoint-cache"
    memo = ValidationMemo.open(root)
    memo.record("record", "sha256:" + "5" * 64)
    memo.flush()
    with memo.path.open("a") as handle:
        handle.write('{"kind": "record", not json\n')
    with memo.path.open("a") as handle:
        handle.write(
            json.dumps(
                {
                    "environment": checkpoint_cache_environment_identity(),
                    "kind": "record",
                    "key": "sha256:" + "6" * 64,
                }
            )
            + "\n"
        )

    reopened = ValidationMemo.open(root)
    # Fail toward live: nothing from the corrupt store is trusted.
    assert not reopened.seen("record", "sha256:" + "5" * 64)
    assert not reopened.seen("record", "sha256:" + "6" * 64)
    assert any(
        diagnostic.get("reason") == "checkpoint-cache-corrupt"
        for diagnostic in reopened.diagnostics
    )


def test_validation_memo_tolerates_torn_tail(tmp_path: Path) -> None:
    """A torn final line drops only the tail, keeping durable prefix rows."""

    root = tmp_path / "derived" / "checkpoint-cache"
    memo = ValidationMemo.open(root)
    memo.record("record", "sha256:" + "7" * 64)
    memo.flush()
    with memo.path.open("a") as handle:
        handle.write('{"environment": "sha256:')  # no newline: torn append

    reopened = ValidationMemo.open(root)
    assert reopened.seen("record", "sha256:" + "7" * 64)
    assert any(
        diagnostic.get("reason") == "checkpoint-cache-torn-tail"
        for diagnostic in reopened.diagnostics
    )


def test_environment_identity_binds_schema_bytes_and_epoch() -> None:
    """The environment identity is a digest over schemas, library, and epoch."""

    identity = checkpoint_cache_environment_identity()
    assert identity.startswith("sha256:")
    assert identity == checkpoint_cache_environment_identity()
    assert isinstance(CHECKPOINT_CACHE_EPOCH, str) and CHECKPOINT_CACHE_EPOCH


# ---------------------------------------------------------------------------
# Memoized full validation: equivalence, honesty, cold start
# ---------------------------------------------------------------------------


def test_memoized_validation_skips_schema_work_and_matches_live(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A warm memo removes jsonschema re-validation without changing results."""

    campaign = _campaign(tmp_path)
    _commit(campaign, "m_zoo0", b"zoo0 bytes")
    _commit(campaign, "m_zoo1", b"zoo1 bytes")
    baseline = _full(campaign)

    counter = _SchemaValidationCounter(monkeypatch)
    memo_root = tmp_path / "derived" / "checkpoint-cache"
    memo = ValidationMemo.open(memo_root)
    cold = _full(campaign, memo)
    cold_validations = counter.take()
    memo.flush()
    assert cold == baseline
    assert cold_validations > 0

    warm_memo = ValidationMemo.open(memo_root)
    warm = _full(campaign, warm_memo)
    warm_validations = counter.take()
    assert warm == baseline
    assert warm_validations == 0


def test_memo_never_excuses_record_content_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A memoized record whose bytes drift on disk still refuses live."""

    campaign = _campaign(tmp_path)
    _commit(campaign, "m_zoo0", b"zoo0 bytes")
    memo_root = tmp_path / "derived" / "checkpoint-cache"
    memo = ValidationMemo.open(memo_root)
    _full(campaign, memo)
    memo.flush()

    shard = campaign["ledger"].path
    lines = shard.read_text().splitlines()
    mutated = json.loads(lines[0])
    mutated["work_id"] = "work-forged"
    lines[0] = json.dumps(mutated, sort_keys=True, separators=(",", ":"))
    shard.write_text("\n".join(lines) + "\n")

    warm_memo = ValidationMemo.open(memo_root)
    with pytest.raises(LedgerCorruptionError):
        _full(campaign, warm_memo)


def test_memo_never_excuses_reconstruction_byte_drift(tmp_path: Path) -> None:
    """A memoized document whose committed bytes change quarantines live."""

    campaign = _campaign(tmp_path)
    _commit(campaign, "m_zoo0", b"zoo0 bytes")
    memo_root = tmp_path / "derived" / "checkpoint-cache"
    memo = ValidationMemo.open(memo_root)
    clean = _full(campaign, memo)
    memo.flush()
    assert len(clean.transactions) == 1 and not clean.stale_transactions

    reconstruction_dir = campaign["canonical_root"] / "reconstruction"
    committed = sorted(reconstruction_dir.rglob("*.json"))
    assert committed
    committed[0].write_text(committed[0].read_text().replace("zoo0", "zoo9"))

    warm_memo = ValidationMemo.open(memo_root)
    drifted = _full(campaign, warm_memo)
    assert not drifted.transactions
    assert len(drifted.stale_transactions) == 1
    (stale,) = drifted.stale_transactions.values()
    assert "reconstruction" in stale.reason


def test_deleting_the_memo_reproduces_uncached_results(tmp_path: Path) -> None:
    """The memo is a derived artifact: deletion falls back to today exactly."""

    campaign = _campaign(tmp_path)
    _commit(campaign, "m_zoo0", b"zoo0 bytes")
    memo_root = tmp_path / "derived" / "checkpoint-cache"
    memo = ValidationMemo.open(memo_root)
    warmed = _full(campaign, memo)
    memo.flush()
    assert memo.path.exists()
    memo.path.unlink()

    cold = _full(campaign, ValidationMemo.open(memo_root))
    plain = _full(campaign)
    assert cold == plain == warmed


# ---------------------------------------------------------------------------
# Incremental advance: equivalence zoo, O(delta), refusals
# ---------------------------------------------------------------------------


def _start_increment(campaign: dict[str, Any]) -> ArtifactCheckpointIncrement:
    """Start incremental checkpointing from the on-disk shard.

    Parameters
    ----------
    campaign:
        Campaign world.

    Returns
    -------
    ArtifactCheckpointIncrement
        Fully validated fold state.
    """

    return ArtifactCheckpointIncrement.start(
        (campaign["ledger"].path,), **_checkpoint_kwargs(campaign)
    )


def test_incremental_advance_matches_full_validation_across_shapes(tmp_path: Path) -> None:
    """Every record-shape transition folds to the same projection as full."""

    campaign = _campaign(tmp_path)
    ledger = campaign["ledger"]

    # Generation A: one committed transaction plus one pending staged root.
    _commit(campaign, "m_zoo0", b"zoo0 bytes")
    pending = _stage_only(campaign, "m_zoo1", b"zoo1 bytes")
    increment = _start_increment(campaign)
    generation = ledger.event_count
    assert increment.generation == generation
    assert increment.projection == _full(campaign)

    # Generation B: the pending transaction finalizes across the boundary and
    # a fresh pending root appears.
    authorization, gate_item = _authorize(campaign, pending)
    _finalize(campaign, pending, authorization, gate_item)
    _stage_only(campaign, "m_zoo2", b"zoo2 bytes")
    advanced = increment.advance(ledger.events_since(generation))
    generation = ledger.event_count
    assert increment.generation == generation
    assert advanced == _full(campaign)
    assert ("m_zoo1", "work-m_zoo1") in {(key[0], key[1]) for key in advanced.transactions}

    # Generation C: an interrupted finalization (staged + authorized only)
    # and a schema-stale historical commit.
    interrupted = _stage_only(campaign, "m_zoo3", b"zoo3 bytes")
    _authorize(campaign, interrupted)
    _commit_schema_stale(campaign, "m_stale", b"stale bytes")
    advanced = increment.advance(ledger.events_since(generation))
    full = _full(campaign)
    assert advanced == full
    stale_ids = {key[0] for key in full.stale_transactions}
    assert {"m_zoo3", "m_stale"} <= stale_ids


def test_incremental_advance_validates_only_the_delta(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Per-advance schema work is flat while history keeps growing."""

    campaign = _campaign(tmp_path)
    ledger = campaign["ledger"]
    _commit(campaign, "m_zoo0", b"zoo0 bytes")
    increment = _start_increment(campaign)
    generation = ledger.event_count

    counter = _SchemaValidationCounter(monkeypatch)
    per_advance: list[int] = []
    for index in range(1, 6):
        _commit(campaign, f"m_zoo{index}", f"zoo{index} bytes".encode())
        counter.take()  # Write-side machinery costs are pinned separately below.
        increment.advance(ledger.events_since(generation))
        generation = ledger.event_count
        per_advance.append(counter.take())

    # O(delta): the last one-transaction advance costs exactly what the first
    # did, despite five generations of accumulated history.
    assert per_advance[-1] == per_advance[0]

    # And the advance is strictly cheaper than one full validation of the
    # grown history.
    _full(campaign)
    full_validations = counter.take()
    assert per_advance[-1] < full_validations


def test_memo_bound_ledger_appends_validate_only_the_delta(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The append-time whole-ledger tripwire stops re-validating history.

    ``append_event`` re-runs full chain validation after every append; with
    the memo bound, that pass keeps running in full while jsonschema work
    stays O(new events), so per-commit cost is flat as the ledger grows.
    """

    campaign = _campaign(tmp_path)
    memo = ValidationMemo.open(tmp_path / "derived" / "checkpoint-cache")
    campaign["ledger"].bind_validation_memo(memo)

    counter = _SchemaValidationCounter(monkeypatch)
    per_commit: list[int] = []
    for index in range(6):
        _commit(campaign, f"m_zoo{index}", f"zoo{index} bytes".encode())
        per_commit.append(counter.take())

    assert per_commit[-1] == per_commit[0]


def test_incremental_advance_keeps_global_registry_refusals(tmp_path: Path) -> None:
    """A delta replaying an already-folded event refuses like full validation."""

    campaign = _campaign(tmp_path)
    ledger = campaign["ledger"]
    _commit(campaign, "m_zoo0", b"zoo0 bytes")
    increment = _start_increment(campaign)
    replayed = ledger.events_since(0)[:1]
    with pytest.raises(ArtifactTransitionError):
        increment.advance(replayed)


def test_incremental_advance_refuses_generation_regression(tmp_path: Path) -> None:
    """Folding is append-only: an empty or stale delta is a typed refusal."""

    campaign = _campaign(tmp_path)
    _commit(campaign, "m_zoo0", b"zoo0 bytes")
    increment = _start_increment(campaign)
    before = increment.generation
    advanced = increment.advance(())
    assert increment.generation == before
    assert advanced == increment.projection


# ---------------------------------------------------------------------------
# staged-root lookup: O(1) index instead of a full-ledger deepcopy per model
# ---------------------------------------------------------------------------


def test_staged_lookup_never_copies_the_full_ledger(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Recovering one staged root stops materializing the whole history."""

    campaign = _campaign(tmp_path)
    ledger = campaign["ledger"]
    _commit(campaign, "m_zoo0", b"zoo0 bytes")
    pending = _stage_only(campaign, "m_zoo1", b"zoo1 bytes")

    def forbidden(self: ArtifactEventLedger) -> tuple:
        raise AssertionError("staged-root lookup must not deepcopy the full ledger")

    monkeypatch.setattr(ArtifactEventLedger, "events", property(forbidden))
    found = staged_artifact_for_result(
        ledger,
        stable_id="m_zoo1",
        work_id="work-m_zoo1",
        author_result_id=pending["result"]["result_id"],
    )
    assert found is not None and found.transaction_id == pending["staged"].transaction_id
    assert (
        staged_artifact_for_result(
            ledger,
            stable_id="m_zoo1",
            work_id="work-m_zoo1",
            author_result_id="sha256:" + "0" * 64,
        )
        is None
    )
