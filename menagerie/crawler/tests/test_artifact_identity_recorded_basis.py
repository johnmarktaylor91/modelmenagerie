"""Rehydrated artifacts revalidate at their recorded basis; stale ones re-author.

Rung-12 leg-2 field failure: ``_validate_artifact_identities`` recomputed the
five author-gated identities with the LIVE checker prompt hash on a REHYDRATED
finalized artifact. The wave-8 checker-prompt re-pin therefore made every
pre-rotation ``vet_identity`` mismatch, and the first rehydrated artifact
consulted aborted the whole campaign with ``DriverIntegrationError: author
proposal identity mismatch``.

The contract pinned here (history-vs-live, single freshness authority):

- a REHYDRATED artifact validates at its transaction's own recorded prompt
  pins -- internal consistency of history, at the basis it was accepted under.
  Whether that history is REUSABLE stays the reducer dependency-currency
  projection's sole decision; this check never becomes a second freshness
  authority.
- a FRESH result still validates against the LIVE frozen prompts, and a
  mismatch stays a hard per-model failure (the ``failed:evidence`` lane-failure
  arm pinned by ``test_identity_mismatch_terminal_also_names_its_cause``).
- a rehydrated artifact that fails validation even at its own basis (or whose
  basis is unrecoverable) is a stale CACHE entry: invalidated with the typed
  ``artifact-authority-identity-stale`` event, routed to fresh authorship,
  never a driver crash.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

import menagerie.crawler.driver as driver_module
import menagerie.crawler.driver_admission as driver_admission_module
import menagerie.crawler.driver_models as driver_models_module
import menagerie.crawler.reducer as reducer_module
from menagerie.crawler.driver_admission import (
    ARTIFACT_AUTHORITY_IDENTITY_STALE_DISPOSITION,
    _identity_basis_rotation,
    _live_author_prompt_hash,
    _validate_artifact_identities,
)
from menagerie.crawler.driver_contracts import (
    DriverIntegrationError,
    RecordedIdentityBasis,
)
from menagerie.crawler.recordio import scan_jsonl
from menagerie.crawler.tests.test_slice_f_driver import (
    FakeAuthor,
    _driver,
    _paths,
    _snapshot,
    _test_authority_context,
)

_ROTATED_CHECKER_DIGEST = "sha256:" + "c" * 64


def _crafted_artifact(tmp_path: Path, driver: Any, snapshot: Any, item: Any) -> Any:
    """Author one fully consistent artifact under the CURRENT identity world.

    Parameters
    ----------
    tmp_path, driver, snapshot, item:
        Scratch root, harness driver, intake snapshot, and scheduled item.

    Returns
    -------
    AuthorArtifact
        Valid proposed artifact whose identities bind the current prompts.
    """

    context = _test_authority_context(snapshot, driver.config)
    return FakeAuthor().author(item, tmp_path / "crafted", driver.config, context)


def _rotate_live_checker_digest(monkeypatch: pytest.MonkeyPatch) -> None:
    """Rotate every live checker-prompt digest producer, as a prompt edit would.

    Parameters
    ----------
    monkeypatch:
        Active pytest monkeypatch context.
    """

    for module in (driver_module, reducer_module):
        monkeypatch.setattr(module, "_checker_prompt_hash", lambda: _ROTATED_CHECKER_DIGEST)
    for module in (driver_models_module,):
        monkeypatch.setattr(module, "_current_checker_prompt_hash", lambda: _ROTATED_CHECKER_DIGEST)
    monkeypatch.setattr(
        driver_admission_module,
        "leg_checker_prompt_sha256",
        lambda: _ROTATED_CHECKER_DIGEST,
    )


def test_recorded_basis_validates_history_while_live_basis_hard_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The leg-2 repro: live recompute rejects pre-rotation vets; recorded basis holds.

    Also proves the recorded basis is a tripwire, not a bypass: a wrong
    recorded prompt and a tampered fact tree both still refuse.
    """

    snapshot = _snapshot(tmp_path, count=1)
    driver = _driver(tmp_path, snapshot)
    item = driver._ordered_work(snapshot, {})[0]
    artifact = _crafted_artifact(tmp_path, driver, snapshot, item)
    original_checker = driver_admission_module.leg_checker_prompt_sha256()

    # Control: a fresh artifact binds the live world.
    _validate_artifact_identities(artifact, driver.config, item=item)

    # The checker prompt rotates (wave-8 re-pin). Live-basis validation now
    # rejects the pre-rotation vet identity -- the exact leg-2 crash condition.
    _rotate_live_checker_digest(monkeypatch)
    with pytest.raises(DriverIntegrationError, match="vet_identity"):
        _validate_artifact_identities(artifact, driver.config, item=item)

    # The same artifact validates at its RECORDED basis: history proves
    # internal consistency in the world it was accepted under.
    basis = RecordedIdentityBasis(
        author_prompt_sha256=str(artifact.proposal["author"]["prompt_sha256"]),
        checker_prompt_sha256=original_checker,
    )
    _validate_artifact_identities(artifact, driver.config, item=item, recorded_basis=basis)

    # Tripwire, not bypass: a recorded basis contradicting the proposal's own
    # prompt binding refuses...
    wrong_prompt = replace(basis, author_prompt_sha256="sha256:" + "e" * 64)
    with pytest.raises(DriverIntegrationError, match="recorded author prompt"):
        _validate_artifact_identities(
            artifact, driver.config, item=item, recorded_basis=wrong_prompt
        )
    # ...and a tampered fact tree mismatches the recorded-basis recompute.
    artifact.proposal["proposed_facts"]["external_metadata"]["description"] = "tampered"
    with pytest.raises(DriverIntegrationError, match="identity mismatch"):
        _validate_artifact_identities(artifact, driver.config, item=item, recorded_basis=basis)


def _patch_rehydrate_once(
    monkeypatch: pytest.MonkeyPatch, stable_id: str, crafted: Any, *, stage: bool
) -> None:
    """Serve one crafted canonical artifact for one model, then defer to reality.

    Parameters
    ----------
    monkeypatch:
        Active pytest monkeypatch context.
    stable_id:
        Model whose first canonical-authority consult returns the fixture.
    crafted:
        Artifact standing in for a rehydrated finalized transaction.
    stage:
        Whether to stage the fixture on the reducer ledger first, as the real
        rehydration path guarantees for artifacts it returns.
    """

    real = driver_admission_module.AdmissionEnvironmentMixin._rehydrate_final_authority
    served: set[str] = set()

    def rehydrate(self: Any, item: Any, reducer: Any, operational: Any) -> Any:
        """Return the crafted artifact once for the target model."""

        if item.stable_id == stable_id and stable_id not in served:
            served.add(stable_id)
            if stage:
                return self._stage_author_result(item, crafted, reducer)
            return crafted
        return real(self, item, reducer, operational)

    monkeypatch.setattr(
        driver_admission_module.AdmissionEnvironmentMixin,
        "_rehydrate_final_authority",
        rehydrate,
    )

    # A REAL finalized transaction makes `has_final_event` true, which keeps
    # the author wave pool from speculatively dispatching a session for the
    # model (`_author_session_is_certain` errs toward the serial lane). The
    # fixture ledger is empty, so mirror that gate exactly: any author call
    # observed in these tests is then the serial commit lane's own decision.
    real_certain = driver_admission_module.AdmissionEnvironmentMixin._author_session_is_certain

    def certain(self: Any, item: Any, reducer: Any) -> bool:
        """Mirror the finalized-authority gate for the target model."""

        if item.stable_id == stable_id:
            return False
        return real_certain(self, item, reducer)

    monkeypatch.setattr(
        driver_admission_module.AdmissionEnvironmentMixin,
        "_author_session_is_certain",
        certain,
    )


def test_identity_stale_rehydrated_authority_reauthors_and_discloses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A rehydrated artifact failing identity validation re-authors, never crashes.

    The leg-2 driver-crash shape: canonical authority whose claimed identities
    fail recompute. The entry must be invalidated with the typed
    ``artifact-authority-identity-stale`` event and the model authored fresh;
    the campaign completes.
    """

    snapshot = _snapshot(tmp_path, count=1)
    paths = _paths(tmp_path, snapshot)
    stable_id = snapshot.items[0].stable_id
    fresh_author = FakeAuthor()
    driver = _driver(tmp_path, snapshot, author=fresh_author)
    item = driver._ordered_work(snapshot, {})[0]
    crafted = _crafted_artifact(tmp_path, driver, snapshot, item)
    # The cached entry claims a vet identity its facts do not derive -- the
    # live-basis mismatch every pre-rotation artifact exhibits (its recorded
    # basis is unrecoverable here, so validation falls back to the live world).
    crafted.proposal["vet_identity"] = "sha256:" + "d" * 64
    _patch_rehydrate_once(monkeypatch, stable_id, crafted, stage=False)

    result = driver.run()

    assert result.status == "complete"
    models = scan_jsonl(paths.ledgers.models)
    assert [model["status"]["code"] for model in models] == ["runs"]
    # Fresh authorship happened exactly once; the stale entry was never reused.
    assert fresh_author.calls == {stable_id: 1}
    stale_events = [
        event
        for event in scan_jsonl(paths.operational_ledger)
        if event.get("details", {}).get("disposition")
        == ARTIFACT_AUTHORITY_IDENTITY_STALE_DISPOSITION
    ]
    assert len(stale_events) == 1
    details = stale_events[0]["details"]
    assert details["stable_id"] == stable_id
    assert details["recorded_basis_available"] is False
    assert "vet_identity" in details["violation_summary"]
    assert "fresh authorship" in details["remedy"]


def test_identity_basis_rotation_mirrors_the_reducer_rule(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The rotation gate names exactly the moved axes, and nothing when none moved.

    A rotated checker prompt makes a proposal's embedded vet identity
    irreducible under the live world (the reducer's ``gate-vet``/``model-vet``
    rule), and a rotated author prompt retires the proposal's author binding;
    either forces re-authoring. A matching basis moves nothing -- reuse.
    """

    live_basis = RecordedIdentityBasis(
        author_prompt_sha256=_live_author_prompt_hash(),
        checker_prompt_sha256=driver_admission_module.leg_checker_prompt_sha256(),
    )
    assert _identity_basis_rotation(live_basis) == ()

    old_checker = replace(live_basis, checker_prompt_sha256="sha256:" + "1" * 64)
    moved = _identity_basis_rotation(old_checker)
    assert len(moved) == 1 and moved[0].startswith("checker_prompt ")

    old_author = replace(live_basis, author_prompt_sha256="sha256:" + "2" * 64)
    moved = _identity_basis_rotation(old_author)
    assert len(moved) == 1 and moved[0].startswith("author_prompt ")

    both = RecordedIdentityBasis(
        author_prompt_sha256="sha256:" + "2" * 64,
        checker_prompt_sha256="sha256:" + "1" * 64,
    )
    assert len(_identity_basis_rotation(both)) == 2

    # After a live checker rotation the previously matching basis reads moved:
    # the exact leg-2 world transition.
    _rotate_live_checker_digest(monkeypatch)
    moved = _identity_basis_rotation(live_basis)
    assert len(moved) == 1 and moved[0].startswith("checker_prompt ")
    assert _ROTATED_CHECKER_DIGEST in moved[0]


def test_same_world_rehydrated_authority_is_reused_without_reauthoring(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Canonical authority whose recorded basis still matches the live world reuses.

    The common resume case: nothing rotated, so the rehydrated artifact passes
    recorded-basis validation, is reused end to end (zero author sessions),
    and reduces to a runs record with no invalidation event.
    """

    snapshot = _snapshot(tmp_path, count=1)
    paths = _paths(tmp_path, snapshot)
    stable_id = snapshot.items[0].stable_id
    fresh_author = FakeAuthor()
    driver = _driver(tmp_path, snapshot, author=fresh_author)
    item = driver._ordered_work(snapshot, {})[0]
    crafted = _crafted_artifact(tmp_path, driver, snapshot, item)
    crafted = replace(
        crafted,
        recorded_identity_basis=RecordedIdentityBasis(
            author_prompt_sha256=str(crafted.proposal["author"]["prompt_sha256"]),
        checker_prompt_sha256=driver_admission_module.leg_checker_prompt_sha256(),
        ),
    )
    _patch_rehydrate_once(monkeypatch, stable_id, crafted, stage=True)

    result = driver.run()

    assert result.status == "complete"
    models = scan_jsonl(paths.ledgers.models)
    assert [model["status"]["code"] for model in models] == ["runs"]
    # The consistent recorded artifact was reused: no author session at all.
    assert fresh_author.calls == {}
    stale_events = [
        event
        for event in scan_jsonl(paths.operational_ledger)
        if event.get("details", {}).get("disposition")
        == ARTIFACT_AUTHORITY_IDENTITY_STALE_DISPOSITION
    ]
    assert stale_events == []
