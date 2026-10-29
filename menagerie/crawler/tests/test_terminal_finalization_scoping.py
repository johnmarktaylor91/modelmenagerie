"""Rung-11 regressions: one model's stale dependency vector must not kill a run.

Pilot rung 11 (2026-08-07, main @ ``64608689``): the resume after the mid-campaign
ff-sync reloaded m10949's disposable author-result cache, which was anchored to
its ORIGINAL staged authoring while two later same-work run-modes repair
generations (both refused, both staged) had superseded it on the artifact ledger.
When the repair cap short-circuited (both durable briefs already spent) and the
driver terminalized ``failed:runner``, ``reducer.prepare_model`` derived the
dependency vector from the work-scoped LATEST staged event while the in-memory
artifact carried the cached transaction, and ``append_artifact_authorization``
correctly refused: ``authorization dependency vector is stale``. The refusal --
a fact about ONE model's artifact authority -- was re-raised through
``_TerminalArtifactEscape`` past every handler and aborted the campaign,
stranding m4256, m7305, and m9304.

Two fixes, both proved here through the real driver loop:

1. SCOPING -- ``_terminalize`` contains a finalization refusal model-scoped: the
   same terminal is re-appended with the artifact kept and only finalization
   withheld (``finalize_artifact=False``), the refusal is reported loud on the
   operational ledger, and the record is flagged for human review. Campaign-level
   causes (pauses, retryable transport, provider backoff, programming errors)
   still propagate untouched.
2. RESUME -- the author-result cache reload detects supersession: a cache whose
   result was staged but is no longer the work's latest artifact event is
   discarded (the ledger's ``author-cache-discarded`` machinery) and the model is
   re-authored under the current authority, so the resume finalizes honestly
   instead of replaying the refusal.

The staleness tripwire itself is UNCHANGED: superseded bytes never finalize.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Optional

import pytest

from menagerie.crawler.author_dispatch import serialize_author_result_cache
from menagerie.crawler.authority import AuthorityContext
from menagerie.crawler.driver_contracts import (
    AuthorArtifact,
    DriverConfig,
    DriverPaused,
    RetryableOperatorError,
    WorkItem,
)
from menagerie.crawler.recordio import scan_jsonl
from menagerie.crawler.tests.test_slice_f_driver import (
    FakeAuthor,
    ForwardScript,
    ScriptedForward,
    _driver,
    _model_records,
    _paths,
    _refresh_proposal_identities,
    _rebind_fake_author_result,
    _snapshot,
)


class SameWorkEvalOnlyAuthor(FakeAuthor):
    """The m10949 shape: every generation stays eval-only on the SAME work identity.

    Each call returns a byte-distinct result (the author-owned ``proposal_id``
    varies) that declares eval-only meaningful modes and keeps
    ``item.active_work_id`` unchanged. When the worker detects ``train`` too,
    every bounded repair generation fails mode coverage, consumes the cap, and
    leaves its refused-but-staged authoring on the artifact ledger under the same
    work -- exactly the pilot's superseding staged events.
    """

    def __init__(self, salt: str = "leg1") -> None:
        """Track captured artifacts per model for cache reconstruction.

        Parameters
        ----------
        salt:
            Per-session byte salt. A REAL agentic re-author never reproduces a
            prior session byte-for-byte, so a resume's fresh authoring stages a
            NEW transaction; the deterministic fake needs the salt to model that.
        """

        super().__init__()
        self.salt = salt
        self.captured: dict[str, list[AuthorArtifact]] = {}

    def author(
        self, item: WorkItem, work_root: Path, config: DriverConfig, context: AuthorityContext
    ) -> AuthorArtifact:
        """Return one more byte-distinct eval-only proposal for the same work."""

        artifact = super().author(item, work_root, config, context)
        facts = artifact.proposal["proposed_facts"]
        facts["modes"]["meaningful_modes"] = ["eval"]
        facts["external_metadata"]["modes"]["meaningful_modes"] = ["eval"]
        artifact.proposal["proposal_id"] = (
            f"proposal-{self.salt}-generation-{self.calls[item.stable_id]}"
        )
        _refresh_proposal_identities(
            artifact.proposal,
            checker_model=config.checker_model,
            checker_version=config.checker_version,
        )
        rebound = _rebind_fake_author_result(artifact)
        self.captured.setdefault(item.stable_id, []).append(rebound)
        return rebound


def _operational_events(tmp_path: Path, snapshot: Any) -> list[dict[str, Any]]:
    """Return every operational event recorded by a synthetic run."""

    return scan_jsonl(_paths(tmp_path, snapshot).operational_ledger)


def _artifact_events(tmp_path: Path, snapshot: Any, stable_id: str) -> list[Mapping[str, Any]]:
    """Return one model's artifact-ledger events in append order."""

    return [
        event
        for event in scan_jsonl(_paths(tmp_path, snapshot).ledgers.artifacts)
        if event.get("stable_id") == stable_id
    ]


@pytest.mark.smoke
def test_stale_dependency_vector_terminalizes_model_scoped(tmp_path: Path) -> None:
    """The rung-11 crash shape lands a model-scoped terminal, not a campaign abort.

    One model's repair generations stage superseding same-work authorings, its
    cap-exhaustion terminal carries the ORIGINAL artifact, and the reducer's
    exactness tripwire refuses finalization. The campaign must complete, the
    victim must hold an honest ``failed:author`` terminal (DX-9/RE-9: repair
    exhaustion is an author-lane fact, never a runner verdict) flagged for human
    review, the refusal must be durable on the operational ledger, and the
    sibling model must be untouched.
    """

    snapshot = _snapshot(tmp_path, count=2)
    victim = snapshot.items[0].stable_id
    sibling = snapshot.items[1].stable_id
    author = SameWorkEvalOnlyAuthor()
    result = _driver(
        tmp_path,
        snapshot,
        author=author,
        forward=ScriptedForward(ForwardScript(expanded_id=victim)),
        run_repair_max=2,
    ).run()

    # Human review is pending for the contained terminal, so the honest
    # completion status is the terminal partition, never a campaign abort.
    assert result.status == "terminal-partition-complete", (
        "one model's stale dependency vector aborted the campaign again"
    )
    models = _model_records(tmp_path, snapshot)
    assert models[victim]["status"]["code"] == "failed:author"
    assert models[victim]["status"]["reason_code"] == "repair-exhausted"
    assert models[victim]["status"]["human_review"]["required"] is True
    assert models[sibling]["status"]["code"] == "runs"
    # Original + two refused repair generations, all on the same work identity.
    staged = [
        event
        for event in _artifact_events(tmp_path, snapshot, victim)
        if event.get("event_kind") == "staged-private"
    ]
    assert len(staged) == 3
    assert len({event["work_id"] for event in staged}) == 1
    assert len({event["author_result_id"] for event in staged}) == 3
    # Finalization was withheld, never worked around: no authorization landed.
    assert not [
        event
        for event in _artifact_events(tmp_path, snapshot, victim)
        if event.get("event_kind") in {"publication-authorized", "terminal-authorized"}
    ]
    refusals = [
        event
        for event in _operational_events(tmp_path, snapshot)
        if event.get("details", {}).get("disposition") == "terminal-finalization-refused"
        and event.get("details", {}).get("stable_id") == victim
    ]
    assert refusals, "the finalization refusal must be durable, not inferred"
    # AW-2's reducer-selected transaction pointer (the m10949 class) now
    # refuses this shape FIRST -- the terminal carries the ORIGINAL artifact
    # while two newer staged roots supersede it as the work's author
    # authority -- with a typed message naming both transactions. The
    # exactness tripwire ("authorization dependency vector is stale") is
    # untouched and still owns non-supersession staleness reached past the
    # pointer.
    assert (
        "not the reducer-selected authority" in refusals[0]["details"]["error"]
        or "authorization dependency vector is stale" in refusals[0]["details"]["error"]
    )


@pytest.mark.smoke
def test_superseded_author_cache_is_discarded_and_resume_completes(tmp_path: Path) -> None:
    """A resume over a superseded cache re-authors and finalizes honestly.

    Leg 1 reproduces the pilot's frozen state: the victim's original authoring is
    staged and cached, two refused same-work repair generations stage later
    authorings, and the leg dies (a campaign pause at the award boundary) before
    any terminal lands. Leg 2 resumes: the stale cache must be DISCARDED as
    superseded, the model re-authored once under current authority, and its
    cap-exhausted ``failed:runner`` terminal recorded -- with finalization now
    consistent (no refusal event on the resume leg).
    """

    snapshot = _snapshot(tmp_path, count=1)
    victim = snapshot.items[0].stable_id
    leg_one_author = SameWorkEvalOnlyAuthor()

    def _kill_at_award(boundary: str, stable_id: str) -> None:
        """Die exactly where rung 11's leg 2 died: before the terminal lands."""

        if boundary == "award-commit-entered" and stable_id == victim:
            raise DriverPaused("injected mid-terminal campaign pause")

    first = _driver(
        tmp_path,
        snapshot,
        author=leg_one_author,
        forward=ScriptedForward(ForwardScript(expanded_id=victim)),
        run_repair_max=2,
        boundary=_kill_at_award,
    ).run()
    assert first.status.startswith("paused")
    assert victim not in _model_records(tmp_path, snapshot)
    staged = [
        event
        for event in _artifact_events(tmp_path, snapshot, victim)
        if event.get("event_kind") == "staged-private"
    ]
    assert len(staged) == 3, "leg 1 must leave the superseding staged authorings"
    # The pilot's stale cache: the ORIGINAL authoring, written before the repair
    # generations staged their superseding results.
    original = leg_one_author.captured[victim][0]
    cache = _paths(tmp_path, snapshot).work_root / victim / "driver-author-artifact.json"
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(
        json.dumps(
            serialize_author_result_cache(
                original.author_result,
                source_manifest=original.source_manifest,
                model_dir=original.model_dir,
            ),
            default=str,
        )
    )

    leg_two_author = SameWorkEvalOnlyAuthor(salt="leg2")
    second = _driver(
        tmp_path,
        snapshot,
        author=leg_two_author,
        forward=ScriptedForward(ForwardScript(expanded_id=victim)),
        run_repair_max=2,
    ).run()

    assert second.status == "terminal-partition-complete"
    discards = [
        event
        for event in _operational_events(tmp_path, snapshot)
        if event.get("details", {}).get("disposition") == "author-cache-discarded"
        and event.get("details", {}).get("stable_id") == victim
    ]
    assert discards, "the superseded cache must be discarded, not replayed"
    assert "superseded" in discards[0]["details"]["error"]
    assert leg_two_author.calls == {}, "a stale cache cannot remint this generation's allowance"
    models = _model_records(tmp_path, snapshot)
    assert models[victim]["status"]["code"] == "failed:author"
    refusal_legs = [
        event
        for event in _operational_events(tmp_path, snapshot)
        if event.get("details", {}).get("disposition") == "terminal-finalization-refused"
    ]
    assert not refusal_legs, "with the cache discarded the resume's terminal finalizes consistently"


@pytest.mark.smoke
def test_superseded_author_cache_resume_reauthors_when_allowance_remains(
    tmp_path: Path,
) -> None:
    """The rung-11 resume still re-authors once when the generation has budget left.

    Same shape as the sibling above with ``run_repair_max=1``, so leg 1 stages the
    original authoring plus ONE refused repair -- two of the generation's three
    chargeable sessions. The resume therefore has allowance remaining: the
    superseded cache is discarded, the model is re-authored EXACTLY once under
    current authority, and the repair cap short-circuits into its ``failed:author``
    repair-exhausted terminal (DX-9/RE-9), finalizing with no refusal. This is the half the derived
    author-session cap must never swallow.
    """

    snapshot = _snapshot(tmp_path, count=1)
    victim = snapshot.items[0].stable_id
    leg_one_author = SameWorkEvalOnlyAuthor()

    def _kill_at_award(boundary: str, stable_id: str) -> None:
        """Die exactly where rung 11's leg 2 died: before the terminal lands."""

        if boundary == "award-commit-entered" and stable_id == victim:
            raise DriverPaused("injected mid-terminal campaign pause")

    first = _driver(
        tmp_path,
        snapshot,
        author=leg_one_author,
        forward=ScriptedForward(ForwardScript(expanded_id=victim)),
        run_repair_max=1,
        boundary=_kill_at_award,
    ).run()
    assert first.status.startswith("paused")
    assert victim not in _model_records(tmp_path, snapshot)
    staged = [
        event
        for event in _artifact_events(tmp_path, snapshot, victim)
        if event.get("event_kind") == "staged-private"
    ]
    assert len(staged) == 2, "leg 1 must leave allowance for exactly one more session"
    original = leg_one_author.captured[victim][0]
    cache = _paths(tmp_path, snapshot).work_root / victim / "driver-author-artifact.json"
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(
        json.dumps(
            serialize_author_result_cache(
                original.author_result,
                source_manifest=original.source_manifest,
                model_dir=original.model_dir,
            ),
            default=str,
        )
    )

    leg_two_author = SameWorkEvalOnlyAuthor(salt="leg2")
    second = _driver(
        tmp_path,
        snapshot,
        author=leg_two_author,
        forward=ScriptedForward(ForwardScript(expanded_id=victim)),
        run_repair_max=1,
    ).run()

    discards = [
        event
        for event in _operational_events(tmp_path, snapshot)
        if event.get("details", {}).get("disposition") == "author-cache-discarded"
        and event.get("details", {}).get("stable_id") == victim
    ]
    assert discards, "the superseded cache must be discarded, not replayed"
    assert leg_two_author.calls == {victim: 1}, "the resume re-authors ONCE under current authority"
    models = _model_records(tmp_path, snapshot)
    assert models[victim]["status"]["code"] == "failed:author"
    assert models[victim]["status"]["reason_code"] == "repair-exhausted"
    refusal_legs = [
        event
        for event in _operational_events(tmp_path, snapshot)
        if event.get("details", {}).get("disposition") == "terminal-finalization-refused"
    ]
    assert not refusal_legs, "with the cache discarded the resume's terminal finalizes consistently"
    assert second.status.startswith(("complete", "terminal-partition-complete"))


@pytest.mark.smoke
def test_missing_cache_resume_at_cap_records_the_terminal_instead_of_parking(
    tmp_path: Path,
) -> None:
    """A pre-attempt cap terminal RECORDS cleanly; it never parks review-held.

    Same frozen leg-1 state as the superseded-cache sibling (three staged
    authorings, the generation's whole allowance), but the disposable cache is
    MISSING on the resume rather than superseded: no supersession discard is
    attested, so nothing is retired and the pre-dispatch valve fires with zero
    author sessions bought and zero attempts of the leg's own. The terminal
    assembly must then name the generation's environment from its authenticated
    durable attempts (never the ``not-applicable`` sentinel that contradicts the
    proof), so the reducer records ``failed:author`` /
    ``effort-exhausted:author-sessions`` instead of refusing it into a
    review-held terminal-recording park.
    """

    snapshot = _snapshot(tmp_path, count=1)
    victim = snapshot.items[0].stable_id
    leg_one_author = SameWorkEvalOnlyAuthor()

    def _kill_at_award(boundary: str, stable_id: str) -> None:
        """Die exactly where rung 11's leg 2 died: before the terminal lands."""

        if boundary == "award-commit-entered" and stable_id == victim:
            raise DriverPaused("injected mid-terminal campaign pause")

    first = _driver(
        tmp_path,
        snapshot,
        author=leg_one_author,
        forward=ScriptedForward(ForwardScript(expanded_id=victim)),
        run_repair_max=2,
        boundary=_kill_at_award,
    ).run()
    assert first.status.startswith("paused")
    staged = [
        event
        for event in _artifact_events(tmp_path, snapshot, victim)
        if event.get("event_kind") == "staged-private"
    ]
    assert len(staged) == 3, "leg 1 must spend the generation's whole allowance"
    cache = _paths(tmp_path, snapshot).work_root / victim / "driver-author-artifact.json"
    cache.unlink(missing_ok=True)

    leg_two_author = SameWorkEvalOnlyAuthor(salt="leg2")
    second = _driver(
        tmp_path,
        snapshot,
        author=leg_two_author,
        forward=ScriptedForward(ForwardScript(expanded_id=victim)),
        run_repair_max=2,
    ).run()

    assert leg_two_author.calls == {}, "no session may be bought past the cap"
    models = _model_records(tmp_path, snapshot)
    assert models[victim]["status"]["code"] == "failed:author"
    assert models[victim]["status"]["reason_code"] == "effort-exhausted:author-sessions"
    refusals = [
        event
        for event in _operational_events(tmp_path, snapshot)
        if event.get("details", {}).get("disposition") == "terminal-recording-refused"
    ]
    assert not refusals, "the cap terminal must record, never park review-held"
    assert second.status == "terminal-partition-complete"


@pytest.mark.smoke
def test_award_authorization_refusal_is_model_scoped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A pre-commit authorization refusal on the RUNS award path stays one model's.

    The award path calls ``_authorize_and_publish_artifact`` before the canonical
    append; a reducer exactness refusal there (nothing durable committed) used to
    propagate to the CLI and abort the campaign. It must terminalize the refused
    model ``failed:runner`` witnessed by its own attempt while the sibling still
    awards ``runs`` -- and post-commit publication failures keep propagating
    (covered by the containment classifier tests above).
    """

    from menagerie.crawler.artifact_transactions import ArtifactBindingError
    from menagerie.crawler.reducer import CanonicalReducer

    snapshot = _snapshot(tmp_path, count=2)
    victim = snapshot.items[0].stable_id
    sibling = snapshot.items[1].stable_id
    original = CanonicalReducer.authorize_publication

    def _refuse_victim(self: Any, model: Mapping[str, Any], *args: Any, **kwargs: Any) -> Any:
        if model.get("stable_id") == victim:
            raise ArtifactBindingError("authorization dependency vector is stale")
        return original(self, model, *args, **kwargs)

    monkeypatch.setattr(CanonicalReducer, "authorize_publication", _refuse_victim)
    result = _driver(tmp_path, snapshot, author=FakeAuthor()).run()

    assert result.status == "terminal-partition-complete"
    models = _model_records(tmp_path, snapshot)
    assert models[victim]["status"]["code"] == "failed:runner"
    assert models[victim]["status"]["reason_code"] == "protocol-violation"
    assert models[victim]["status"]["human_review"]["required"] is True
    assert models[sibling]["status"]["code"] == "runs"
    refusals = [
        event
        for event in _operational_events(tmp_path, snapshot)
        if event.get("details", {}).get("disposition") == "terminal-finalization-refused"
        and event.get("details", {}).get("stable_id") == victim
    ]
    assert refusals, "the terminal-lane refusal must be durable"


@pytest.mark.smoke
def test_campaign_level_finalization_causes_still_propagate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Containment never absorbs pauses, transport retries, or broken programs."""

    from menagerie.crawler import driver as driver_module

    for index, cause in enumerate(
        (RetryableOperatorError("transport blinked"), AttributeError("broken"))
    ):
        root = tmp_path / f"cause-{index}"
        root.mkdir()
        snapshot = _snapshot(root, count=1)
        victim = snapshot.items[0].stable_id

        def _refuse(*args: Any, **kwargs: Any) -> None:
            raise cause  # noqa: B023 -- rebound per loop iteration on purpose

        monkeypatch.setattr(driver_module.CrawlerDriver, "_authorize_terminal_artifact", _refuse)
        crawler = _driver(
            root,
            snapshot,
            author=SameWorkEvalOnlyAuthor(),
            forward=ScriptedForward(ForwardScript(expanded_id=victim)),
            run_repair_max=2,
        )
        raised: Optional[BaseException] = None
        try:
            result = crawler.run()
        except BaseException as exc:  # noqa: BLE001 -- propagation IS the assertion
            raised = exc
        # Campaign-level causes surface as the campaign's failure, never as a
        # quietly contained model terminal.
        if raised is None:
            assert result.status != "complete", f"{type(cause).__name__} must stay campaign-scoped"
        else:
            assert isinstance(raised, type(cause))
        refusals = [
            event
            for event in _operational_events(root, snapshot)
            if event.get("details", {}).get("disposition") == "terminal-finalization-refused"
        ]
        assert not refusals, "campaign-level causes must never be contained"
