"""End-to-end: a checker-prompt digest rotation stales completed work at admission.

Wave-8 cross-striker pin (Sol adversarial review). Striker A's record-anchored
quarantine fallback classifies ``checker_prompt`` drift as PERMISSIBLE at the
helper boundary: ``_quarantine_code_identity_drift`` names it a moved
code-identity component rather than record drift, and a unit test exercises
exactly that classifier. Read alone, that test suggests a checker-prompt
rotation silently re-blesses run awards adjudicated under the OLD checker.

End-to-end the behavior is the opposite, and this module pins it. Quarantine
acceptance first requires the model to appear in ``reducer.current_records``
(the quarantine valve in ``driver_admission``), and that dependency-current
projection replays every record's pinned ``checker_prompt_identity`` axis
against the LIVE authority context. A rotated checker digest therefore makes
the old run record NON-current before the record-anchored fallback ever runs:
the completed work is never accepted across the rotation, no acceptance event
is written, the historical record survives un-superseded, and the work parks
in the environment-rebuild deferral ledger for an honest re-drive under the
new checker.

The contrast matters operationally: striker C rotates the checker prompt
digest (wave 8 re-pinned ``codex_accuracy_checker_v2.txt``), and the safety
question is whether awards gated by the old prompt can silently survive. They
cannot -- the helper's whitelist is reachable only for identity axes the
dependency projection does not police, and the checker axis is policed.

COMPOSE (AW x WS-1): the launch itself is now gated too. A projection-stale
runs award refuses an unacknowledged launch at the award-custody door, and
ordinary mode no longer auto-readmits pending-metadata records -- the resume
across the rotation is the operator's ``acknowledge_award_holds`` plus the
NAMED ``--gate-catchup-targets`` re-validation (clause 4 / V3).
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
    QUARANTINE_RECORD_ANCHORED_ACCEPTANCE_DISPOSITION,
    _quarantine_code_identity_drift,
)
from menagerie.crawler.census_ack import award_holds_ack_digest
from menagerie.crawler.driver_contracts import DriverIntegrationError
from menagerie.crawler.env_lifecycle import DiskRecoveryError
from menagerie.crawler.envs import EnvironmentIntent
from menagerie.crawler.recordio import scan_jsonl
from menagerie.crawler.tests.test_slice_f_driver import (
    FakeChecker,
    FakeEnvironments,
    FakePauseScheduler,
    _driver,
    _paths,
    _snapshot,
)

_ROTATED_CHECKER_DIGEST = "sha256:" + "c" * 64


class _TeardownFailingEnvironments(FakeEnvironments):
    """Complete model use and then report a teardown disk failure."""

    def run(self, intent: EnvironmentIntent, *, use: Any) -> object:
        """Invoke use successfully before raising a teardown exception.

        Parameters
        ----------
        intent:
            Environment intent under teardown.
        use:
            Model-use callback executed before the synthetic failure.

        Returns
        -------
        object
            Never returns; the teardown failure always raises.

        Raises
        ------
        DiskRecoveryError
            Always, after the use callback completes.
        """

        super().run(intent, use=use)
        raise DiskRecoveryError("synthetic post-use teardown failure")


def _rotate_live_checker_digest(monkeypatch: pytest.MonkeyPatch) -> None:
    """Rotate every live checker-prompt digest producer, as a prompt edit would.

    A real rotation changes the prompt FILE bytes, so every module-level
    producer reading them moves together; records keep the digest they pinned.

    Parameters
    ----------
    monkeypatch:
        Active pytest monkeypatch context.
    """

    for module in (driver_module, reducer_module, driver_admission_module):
        monkeypatch.setattr(module, "_checker_prompt_hash", lambda: _ROTATED_CHECKER_DIGEST)
    monkeypatch.setattr(
        driver_models_module, "_current_checker_prompt_hash", lambda: _ROTATED_CHECKER_DIGEST
    )
    # BL-10 (e): driver_admission's seams consult the LEG PIN, not the live
    # reader; a cross-leg rotation means the NEW leg pins the rotated digest.
    monkeypatch.setattr(
        driver_admission_module, "leg_checker_prompt_sha256", lambda: _ROTATED_CHECKER_DIGEST
    )


def test_checker_digest_rotation_blocks_record_anchored_acceptance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Dependency projection stales the old award before A's fallback can bless it.

    Mirrors ``test_quarantine_code_identity_hotfix_preserves_completed_work``
    with ONE moved identity swapped: the checker prompt digest instead of the
    award closure. The hotfix variant is accepted on record identities; the
    checker variant must never be, because the record's dependency vector pins
    ``checker_prompt_identity`` and the reducer's currency replay polices it.
    """

    snapshot = _snapshot(tmp_path, count=1)
    paths = _paths(tmp_path, snapshot)
    stable_id = snapshot.items[0].stable_id
    environments = _TeardownFailingEnvironments(tmp_path / "fake-envs")
    pause_scheduler = FakePauseScheduler(paths.wakeup_root)
    first = _driver(
        tmp_path,
        snapshot,
        checker=FakeChecker(quota=True),
        environments=environments,
        pause_scheduler=pause_scheduler,
    ).run()
    assert first.status == "paused:usage-limit"
    models = scan_jsonl(paths.ledgers.models)
    assert [model["status"]["code"] for model in models] == ["runs"]
    baseline_revision = models[0]["record_revision"]

    # The checker prompt rotates between the quarantine snapshot and the
    # resume; every record keeps the digest it pinned.
    _rotate_live_checker_digest(monkeypatch)

    # Spy on the fallback classifier: the pin is not merely that acceptance
    # does not happen, but that the dependency projection stales the record
    # UPSTREAM, so the record-anchored fallback is never consulted at all.
    classifier_calls: list[tuple[Any, Any]] = []
    real_classifier = driver_admission_module._quarantine_code_identity_drift

    def spying_classifier(recorded: Any, live: Any) -> Any:
        """Record every fallback-classifier consultation before delegating."""

        classifier_calls.append((recorded, live))
        return real_classifier(recorded, live)

    monkeypatch.setattr(
        driver_admission_module, "_quarantine_code_identity_drift", spying_classifier
    )

    # COMPOSE (AW x WS-1, RULING 1 + clause 4 / V3): the rotation stales the
    # pending-metadata runs award, so a blind ordinary resume refuses at the
    # award-custody door -- it can never silently funnel the settled award
    # back into the re-drive treadmill.
    with pytest.raises(DriverIntegrationError, match="projection-stale runs award"):
        _driver(
            tmp_path,
            snapshot,
            checker=FakeChecker(quota=True),
            environments=environments,
            pause_scheduler=pause_scheduler,
        ).run()

    # The sanctioned resume is the operator's decision, twice over: the
    # award-holds census is acknowledged after reading it, and the
    # pending-metadata record is re-validated through the NAMED gate-catchup
    # mode (ordinary mode no longer auto-readmits pending-metadata records).
    # Neither lever re-blesses anything: acceptance stays impossible across
    # the rotation.
    statuses = []
    for _ in range(4):
        acked = _driver(
            tmp_path,
            snapshot,
            checker=FakeChecker(quota=True),
            environments=environments,
            pause_scheduler=pause_scheduler,
        )
        acked.config = replace(
            acked.config,
            acknowledge_award_holds=True,
            # D-47: the acknowledgment binds the exact census it read — here
            # the one rotation-staled award and zero live award parks.
            award_holds_digest=award_holds_ack_digest([stable_id], []),
            gate_catchup_targets=(stable_id,),
        )
        result = acked.run()
        statuses.append(result.status)
        if result.status not in {"paused:usage-limit"}:
            break
    assert statuses[-1] == "complete-with-deferrals"
    assert classifier_calls == []

    # 1. The record-anchored fallback NEVER accepts across a checker rotation:
    #    no acceptance event exists, for this or any model.
    canonical_events = scan_jsonl(
        driver_module.canonical_operational_ledger_path(paths.ledgers.models)
    )
    acceptances = [
        event
        for event in canonical_events
        if event.get("details", {}).get("disposition")
        == QUARANTINE_RECORD_ANCHORED_ACCEPTANCE_DISPOSITION
    ]
    assert acceptances == []
    #    The award hold itself was disclosed where operators look: at least
    #    one degraded award-holds-census names the stale award.
    holds = [
        event
        for event in canonical_events
        if event.get("details", {}).get("disposition") == "award-holds-census"
    ]
    assert holds
    assert all(event["details"]["stale_awards"] == [stable_id] for event in holds)

    # 2. History is preserved, not relitigated: the completed run award stays
    #    in the ledger exactly as recorded -- one record, still runs, never
    #    superseded, never terminalized. It is merely non-current.
    models = scan_jsonl(paths.ledgers.models)
    assert len(models) == 1
    assert models[0]["status"]["code"] == "runs"
    assert models[0]["record_revision"] == baseline_revision

    # 3. The outcome is loud and honest: the stale-under-new-checker work parks
    #    in the durable environment-rebuild deferral ledger for a re-drive; no
    #    silent re-blessing and no false build terminal.
    records_root = paths.ledgers.models.parent.parent
    rebuild_ledger = records_root / "intake-extensions" / "environment-rebuild-deferrals.jsonl"
    rows = scan_jsonl(rebuild_ledger) if rebuild_ledger.exists() else []
    assert [row["stable_id"] for row in rows] == [stable_id]
    assert rows[0]["disposition"] == "deferred:environment-rebuild-required"
    attempts = scan_jsonl(paths.ledgers.attempts)
    assert not [
        attempt
        for attempt in attempts
        if attempt.get("failure", {}).get("reason_code") == "build-failed"
    ]


def test_helper_boundary_alone_would_permit_checker_drift() -> None:
    """The classifier the unit tests exercise DOES whitelist checker drift.

    This is the deliberate contrast with the end-to-end test above: the
    permissiveness lives only at the helper boundary, and the reducer's
    dependency-current projection is what actually polices the checker axis.
    If this assertion ever starts failing, the helper tightened and the
    end-to-end pin should be revisited together with striker A's tests.
    """

    recorded = {
        "stable_id": "m_example",
        "proposal_sha256": "sha256:" + "a" * 64,
        "checker_prompt": "sha256:" + "b" * 64,
    }
    live = dict(recorded)
    live["checker_prompt"] = _ROTATED_CHECKER_DIGEST
    assert _quarantine_code_identity_drift(recorded, live) == ("checker_prompt",)
