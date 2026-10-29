"""The H-5 K=2 door's handoff exemption keys on provenance, never config.

P5 opus F4 / fable F11: ``driver.py``'s publication belt exempted any
``ActivatedHandoffArtifact`` from the K=2 duplicate-accept door, but that class
is assigned purely from ``config.only_status`` (driver_admission) and
``handoff_sha256`` defaults to ``""`` -- so a SAME-campaign ``--only-status``
re-run (the host-capability memory cohort, whose valve by construction never
published a handoff) could publish without ever passing the K=2 door. The
remedy keys the exemption on what the real cross-campaign handoff path
actually stamps: a ledger-backed final transaction whose reconstruction inputs
carry a ``handoff_execution`` with a matching non-empty ``handoff_sha256``,
recorded under the ``DEFER_RECOMMENDATION`` arm (the one arm minted ONLY as a
terminal deferral of its origin campaign and consumed only by the routed
re-run -- the provenance spelling of "origin campaign distinct from the
current campaign"). Era honesty: this is a refusal-point tightening on LIVE
publication only; recorded history is untouched.
"""

from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import pytest

import menagerie.crawler.driver as driver_module
from menagerie.crawler.author_dispatch import ProposedAuthorResult
from menagerie.crawler.driver import (
    CrawlerDriver,
    _handoff_inherited_acceptance_authority,
)
from menagerie.crawler.driver_contracts import (
    ActivatedHandoffArtifact,
    AuthorArtifact,
    DriverIntegrationError,
)

_HANDOFF_SHA = "sha256:" + "d" * 64


def _artifact(
    *,
    handoff_sha256: str = "",
    activated: bool = True,
    staged: bool = True,
    author_result: Any = None,
) -> AuthorArtifact:
    """Build one minimally shaped artifact for the provenance predicate."""

    binding = SimpleNamespace(stable_id="m1", work_id="work-m1")
    kwargs: dict[str, Any] = {
        "author_result": author_result
        if author_result is not None
        else SimpleNamespace(binding=binding),
        "source_manifest": {},
        "model_dir": Path("/nonexistent"),
        "staged": SimpleNamespace(transaction_id="txn-1") if staged else None,
    }
    if activated:
        return ActivatedHandoffArtifact(handoff_sha256=handoff_sha256, **kwargs)
    return AuthorArtifact(**kwargs)


def _transaction(
    *,
    handoff_execution: Any,
    recorded_kind: str,
) -> SimpleNamespace:
    """One fake final-transaction projection carrying only the read fields."""

    return SimpleNamespace(
        reconstruction_inputs=SimpleNamespace(
            handoff_execution=handoff_execution,
            author_result={"kind": recorded_kind},
        )
    )


def _index(transaction: Any) -> dict[tuple[str, str, Any], Any]:
    return {("m1", "work-m1", "txn-1"): transaction}


# ---------------------------------------------------------------------------
# The provenance predicate itself.
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_the_genuine_cross_campaign_handoff_shape_is_exempt() -> None:
    transaction = _transaction(
        handoff_execution={"handoff_sha256": _HANDOFF_SHA},
        recorded_kind="DEFER_RECOMMENDATION",
    )
    artifact = _artifact(handoff_sha256=_HANDOFF_SHA)
    assert _handoff_inherited_acceptance_authority(artifact, _index(transaction))


@pytest.mark.smoke
def test_a_config_keyed_handoff_free_artifact_is_not_exempt() -> None:
    # The finding's exact actor: an ActivatedHandoffArtifact constructible
    # from ``config.only_status`` alone, carrying NO handoff (the default
    # ``handoff_sha256 == ""``). The K=2 door applies.
    transaction = _transaction(handoff_execution=None, recorded_kind="PROPOSED")
    artifact = _artifact(handoff_sha256="")
    assert not _handoff_inherited_acceptance_authority(artifact, _index(transaction))


@pytest.mark.smoke
def test_a_same_campaign_proposed_record_is_not_exempt() -> None:
    # A same-campaign --only-status resume rehydrates its OWN PROPOSED record:
    # its acceptance authority is this campaign's, so the door applies even if
    # the artifact somehow carries a sha stamp.
    transaction = _transaction(
        handoff_execution={"handoff_sha256": _HANDOFF_SHA},
        recorded_kind="PROPOSED",
    )
    artifact = _artifact(handoff_sha256=_HANDOFF_SHA)
    assert not _handoff_inherited_acceptance_authority(artifact, _index(transaction))


@pytest.mark.smoke
def test_every_missing_provenance_conjunct_fails_closed() -> None:
    genuine = _transaction(
        handoff_execution={"handoff_sha256": _HANDOFF_SHA},
        recorded_kind="DEFER_RECOMMENDATION",
    )
    # Not an activated handoff at all.
    assert not _handoff_inherited_acceptance_authority(
        _artifact(activated=False), _index(genuine)
    )
    # No staged custody -> no transaction key to resolve.
    assert not _handoff_inherited_acceptance_authority(
        _artifact(handoff_sha256=_HANDOFF_SHA, staged=False), _index(genuine)
    )
    # No ledger-backed final transaction resolves.
    assert not _handoff_inherited_acceptance_authority(
        _artifact(handoff_sha256=_HANDOFF_SHA), {}
    )
    # Constructor stamp disagrees with the ledger-backed handoff digest.
    assert not _handoff_inherited_acceptance_authority(
        _artifact(handoff_sha256="sha256:" + "e" * 64), _index(genuine)
    )
    # Transaction exists but records no handoff execution.
    assert not _handoff_inherited_acceptance_authority(
        _artifact(handoff_sha256=_HANDOFF_SHA),
        _index(_transaction(handoff_execution=None, recorded_kind="DEFER_RECOMMENDATION")),
    )


# ---------------------------------------------------------------------------
# The door itself: a handoff-free activated artifact is refused at K=2 without
# a settled witness; the genuine shape skips the door.
# ---------------------------------------------------------------------------


def _accepted_metadata_gate() -> dict[str, Any]:
    return {
        "gate_id": "gate-metadata-1",
        "gate_kind": "metadata_batch",
        "items": [
            {
                "stable_id": "m1",
                "verdict": "accurate",
                "integrity": {"verdict": "accurate"},
                "rung_check": {"verdict": "accurate"},
            }
        ],
    }


def _proposed_result() -> ProposedAuthorResult:
    binding = SimpleNamespace(stable_id="m1", work_id="work-m1")
    return ProposedAuthorResult(
        binding,  # type: ignore[arg-type]
        {"stable_id": "m1", "work_id": "work-m1"},
        None,  # type: ignore[arg-type]
    )


def _door(
    tmp_path: Path,
    artifact: AuthorArtifact,
    transactions: Mapping[tuple[str, str, Any], Any],
    monkeypatch: pytest.MonkeyPatch,
) -> list[str]:
    """Run ``_authorize_terminal_artifact`` up to the K=2 belt on a fake self."""

    gate = _accepted_metadata_gate()
    monkeypatch.setattr(driver_module, "_find_gate", lambda *a, **k: gate)
    published: list[str] = []
    fake_self = SimpleNamespace(
        _final_artifact_transactions=dict(transactions),
        paths=SimpleNamespace(
            ledgers=SimpleNamespace(
                models=tmp_path / "records" / "models" / "current-shard.jsonl"
            )
        ),
        _authorize_and_publish_artifact=lambda *a, **k: published.append("published"),
    )
    reducer = SimpleNamespace(gate_records=())
    CrawlerDriver._authorize_terminal_artifact(
        fake_self,  # type: ignore[arg-type]
        artifact,
        {"stable_id": "m1"},
        [gate],
        reducer,  # type: ignore[arg-type]
    )
    return published


def test_a_handoff_free_only_status_resume_is_refused_at_the_k2_door(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # RED shape (opus F4): before the fix this artifact -- activated purely by
    # config, no handoff provenance, same-campaign PROPOSED record -- skipped
    # the belt and published with zero settled K=2 witness.
    artifact = _artifact(handoff_sha256="", author_result=_proposed_result())
    transactions = _index(_transaction(handoff_execution=None, recorded_kind="PROPOSED"))
    with pytest.raises(DriverIntegrationError, match="no settled K=2"):
        _door(tmp_path, artifact, transactions, monkeypatch)


def test_the_genuine_handoff_shape_still_skips_the_k2_door(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact = _artifact(handoff_sha256=_HANDOFF_SHA, author_result=_proposed_result())
    transactions = _index(
        _transaction(
            handoff_execution={"handoff_sha256": _HANDOFF_SHA},
            recorded_kind="DEFER_RECOMMENDATION",
        )
    )
    published = _door(tmp_path, artifact, transactions, monkeypatch)
    assert published == ["published"]


@pytest.mark.smoke
def test_the_publication_belt_consults_the_provenance_predicate() -> None:
    source = inspect.getsource(CrawlerDriver._authorize_terminal_artifact)
    assert "_handoff_inherited_acceptance_authority(" in source
    # The old config-keyed guard shape is gone from the belt.
    assert "if not isinstance(artifact, ActivatedHandoffArtifact):" not in source
