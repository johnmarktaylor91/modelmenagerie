"""AB-30 matrix: the Wave-2 batch's PURE closed-set widenings, consolidated.

A pure widening is the one schema change with no reader flip: a closed
vocabulary gains members while every pre-batch member -- and therefore every
pre-batch row spelled in that vocabulary -- stays lawful. This module is the
declared TABLE of those cells for the Wave-2 rotation batch. Each cell
asserts two directions on the REAL production set object (never a re-derived
copy):

* the batch's new member IS in the set (the widening landed), and
* the named pre-batch members are STILL in the set (era honesty -- a
  prior-era row spelled with an old member keeps parsing under the same
  meaning).

Negative controls pin the closure direction: an unreviewed spelling is NOT a
member, and the closed validators refuse it TYPED, so "widened" can never
silently read as "open".

Cells (constant -> new member(s)):

1.  ``constants.OperationalEventKind``            -> ``leg-outcome`` (W6/OB-M12)
2.  ``constants.InvocationOrigin``                -> ``supervised-restart`` (W6)
3.  ``rotation_registry.ROTATION_CLASSES``        -> ``disclosed-non-staling``
4.  ``checker_lane.CHECKER_LANE_FAILURE_CLASSES`` (+ non-ladder subset)
                                                  -> ``checker-verdict-conflict`` (AG-4)
5.  ``scheduler_parks.SCHEDULER_QUARANTINE_REASONS``
                                                  -> W9/AT-4/AT-5 park reasons
6.  ``status.GRANT_RELEASE_CENSUS_CLASSES``       -> four ``machine-released:<class>``
                                                     arms (W14, signed)
7.  ``status.MACHINE_DEFERRAL_DISPOSITIONS``      -> quota-storm valve + recidivist hold
8.  ``paid_session_accounting.ATTEMPT_REFUND_LOCAL_OUTCOMES``
                                                  -> ``author-blocked-prerequisite``
                                                     (AL-6; NOT infrastructure)
9.  ``tools.launch_preflight.EXPECTED_PREFLIGHT_ARMS``
                                                  -> ``checker-calibration-mode``
10. leg-event closed kind vocabulary              -> ``leg_started`` / ``leg_ended`` (W6)
11. ``envs.LEGACY_UNBOUND_OVERLAY_MANIFESTS``     -> EN-8 carve-out stays EXACTLY two
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import pytest

from menagerie.crawler.checker_lane import (
    CHECKER_LANE_FAILURE_CLASSES,
    CHECKER_LANE_NON_LADDER_FAILURE_CLASSES,
)
from menagerie.crawler.constants import InvocationOrigin, OperationalEventKind
from menagerie.crawler.env_lifecycle import (
    EnvironmentOverlayError,
    SequentialEnvironmentLifecycle,
)
from menagerie.crawler.envs import (
    LEGACY_UNBOUND_OVERLAY_MANIFESTS,
    compute_environment_generation,
    pip_overlay_export_binding,
    render_pip_overlay_manifest,
)
from menagerie.crawler.identity import hash_bytes, stamp_row_digest
from menagerie.crawler.leg_generation import (
    LEG_EVENT_SCHEMA_VERSION,
    LegGenerationError,
    validate_leg_event_row,
)
from menagerie.crawler.paid_session_accounting import (
    ATTEMPT_REFUND_INFRASTRUCTURE_LOCAL_OUTCOMES,
    ATTEMPT_REFUND_LOCAL_OUTCOMES,
)
from menagerie.crawler.quota_deferral import (
    QUOTA_DEFERRAL_DISPOSITION,
    STORM_RECIDIVIST_DISPOSITION,
)
from menagerie.crawler.release_vocabulary import MACHINE_RELEASE_CLASSES
from menagerie.crawler.rotation_registry import ROTATION_CLASSES
from menagerie.crawler.scheduler_parks import SCHEDULER_QUARANTINE_REASONS
from menagerie.crawler.status import (
    GRANT_RELEASE_CENSUS_CLASSES,
    MACHINE_DEFERRAL_DISPOSITIONS,
)
from menagerie.crawler.tests.mixed_era import assert_era_outcome, classify_reader
from menagerie.crawler.tools.launch_preflight import EXPECTED_PREFLIGHT_ARMS

pytestmark = pytest.mark.smoke

#: A spelling no lane ever proposed: the closure-direction probe. It must be
#: absent from EVERY widened set -- a widening adds reviewed members, it never
#: opens the vocabulary.
_FUTURE_SPELLING = "era-from-the-future"


def _enum_values(enum_cls: Any) -> frozenset[str]:
    """The real value set of a closed StrEnum, without version-dependent ``in``."""

    return frozenset(member.value for member in enum_cls)


# ---------------------------------------------------------------------------
# Cells 1-2: closed StrEnums in constants.py
# ---------------------------------------------------------------------------


def test_operational_event_kind_gains_leg_outcome() -> None:
    """Cell 1: OperationalEventKind widens with ``leg-outcome`` (W6/OB-M12)."""

    values = _enum_values(OperationalEventKind)
    assert "leg-outcome" in values
    # Era honesty: prior-era operational rows stay lawful spellings.
    for prebatch in ("leg-config", "wakeup", "checkpoint", "requeue-grant-consumed"):
        assert prebatch in values, prebatch


def test_invocation_origin_gains_supervised_restart() -> None:
    """Cell 2: InvocationOrigin widens with ``supervised-restart`` (W6 ladder)."""

    values = _enum_values(InvocationOrigin)
    assert "supervised-restart" in values
    for prebatch in ("ordinary-run", "manual-resume", "wake-callback"):
        assert prebatch in values, prebatch


# ---------------------------------------------------------------------------
# Cells 3-5: closed frozenset vocabularies
# ---------------------------------------------------------------------------


def test_rotation_classes_gain_disclosed_non_staling() -> None:
    """Cell 3: ROTATION_CLASSES widens with ``disclosed-non-staling``."""

    assert "disclosed-non-staling" in ROTATION_CLASSES
    for prebatch in ("refinement", "soundness"):
        assert prebatch in ROTATION_CLASSES, prebatch
    # The widened set is exactly the reviewed three: still closed, not open.
    assert ROTATION_CLASSES == frozenset(
        {"refinement", "soundness", "disclosed-non-staling"}
    )


def test_checker_lane_failure_classes_gain_verdict_conflict() -> None:
    """Cell 4: both checker-lane class sets widen with the AG-4 conflict park."""

    assert "checker-verdict-conflict" in CHECKER_LANE_FAILURE_CLASSES
    assert "checker-verdict-conflict" in CHECKER_LANE_NON_LADDER_FAILURE_CLASSES
    for prebatch in (
        "permanent-contract-rejection",
        "partial-result",
        "invalid-gate-result",
        "lane-error",
    ):
        assert prebatch in CHECKER_LANE_FAILURE_CLASSES, prebatch
    for prebatch in ("checker-disclosure-debt", "checker-wall-exhausted"):
        assert prebatch in CHECKER_LANE_NON_LADDER_FAILURE_CLASSES, prebatch
    # The non-ladder set stays a strict subset of the full failure vocabulary.
    assert CHECKER_LANE_NON_LADDER_FAILURE_CLASSES < CHECKER_LANE_FAILURE_CLASSES


def test_scheduler_quarantine_reasons_gain_three_parks() -> None:
    """Cell 5: SCHEDULER_QUARANTINE_REASONS widens with W9 + AT-5 + AT-4."""

    for new in (
        "invalidation-nonconvergent",
        "terminal-proof-generation-exhausted",
        "award-blocked:spent-chain",
    ):
        assert new in SCHEDULER_QUARANTINE_REASONS, new
    for prebatch in (
        "member-fault",
        "non-model-fault",
        "award-unroutable",
        "no-durable-routing-progress",
    ):
        assert prebatch in SCHEDULER_QUARANTINE_REASONS, prebatch


# ---------------------------------------------------------------------------
# Cells 6-7: release census + machine-deferral partition (status.py)
# ---------------------------------------------------------------------------


def test_grant_release_census_gains_the_four_machine_arms() -> None:
    """Cell 6: GRANT_RELEASE_CENSUS_CLASSES widens with W14's signed class arms.

    Waiver doctrine part 4: each machine class is its own segregated
    ``machine-released:<class>`` census arm, never counted as reviewed.
    """

    assert MACHINE_RELEASE_CLASSES == frozenset(
        {
            "quota-deferral",
            "corpus-conflict",
            "blocked-prerequisite-post-rotation",
            "engine-invalidated-artifact-cache",
        }
    )
    for name in sorted(MACHINE_RELEASE_CLASSES):
        assert f"machine-released:{name}" in GRANT_RELEASE_CENSUS_CLASSES, name
    # Era honesty: the pre-batch census arms keep their exact spellings.
    for prebatch in (
        "award-released",
        "review-released",
        "infra-waiver-released",
        "unattributed",
    ):
        assert prebatch in GRANT_RELEASE_CENSUS_CLASSES, prebatch
    # The census stays a duplicate-free closed key tuple.
    assert len(set(GRANT_RELEASE_CENSUS_CLASSES)) == len(GRANT_RELEASE_CENSUS_CLASSES)


def test_machine_deferral_dispositions_gain_the_quota_storm_valve() -> None:
    """Cell 7: MACHINE_DEFERRAL_DISPOSITIONS widens with the storm valve + hold."""

    assert QUOTA_DEFERRAL_DISPOSITION == "machine-deferred:quota-storm"
    assert STORM_RECIDIVIST_DISPOSITION == "review-held:storm-recidivist"
    assert "machine-deferred:quota-storm" in MACHINE_DEFERRAL_DISPOSITIONS
    assert "review-held:storm-recidivist" in MACHINE_DEFERRAL_DISPOSITIONS
    for prebatch in (
        "deferred:needs-environment-coverage",
        "deferred:checker-lane-contract-failure",
        "deferred:scheduler-fault-quarantined",
    ):
        assert prebatch in MACHINE_DEFERRAL_DISPOSITIONS, prebatch


# ---------------------------------------------------------------------------
# Cell 8: attempt-refund outcomes (AL-6 -- a widening that must NOT reach
# the infrastructure subset)
# ---------------------------------------------------------------------------


def test_attempt_refund_gains_author_blocked_outside_infrastructure() -> None:
    """Cell 8: the refund union widens; the infrastructure subset does not.

    AL-6's honest BLOCKED verdict refunds the attempt but is deliberately
    NOT machine weather: the provider-storm fuse's set keeps exactly its
    prior seven members, so no author verdict can ever trip a storm fuse.
    """

    assert "author-blocked-prerequisite" in ATTEMPT_REFUND_LOCAL_OUTCOMES
    # The seven 200a machine classes plus item 9's typed executor outage
    # (2026-09-01, recovery F1): still machine-only, still storm-countable.
    infrastructure_eight = frozenset(
        {
            "campaign-teardown",
            "failed-author-queue-stalled",
            "failed-checker-session-wall",
            "failed-custody-reaped",
            "failed-executor-typed-outage",
            "failed-harness-aborted-streaming-sigterm",
            "failed-harness-api-error",
            "failed-machine-spawn",
        }
    )
    assert ATTEMPT_REFUND_INFRASTRUCTURE_LOCAL_OUTCOMES == infrastructure_eight
    assert "author-blocked-prerequisite" not in (
        ATTEMPT_REFUND_INFRASTRUCTURE_LOCAL_OUTCOMES
    )
    assert ATTEMPT_REFUND_LOCAL_OUTCOMES == infrastructure_eight | {
        "author-blocked-prerequisite"
    }


# ---------------------------------------------------------------------------
# Cell 9: armed-guard preflight census
# ---------------------------------------------------------------------------


def test_preflight_arms_gain_checker_calibration_mode() -> None:
    """Cell 9: EXPECTED_PREFLIGHT_ARMS widens with ``checker-calibration-mode``."""

    assert "checker-calibration-mode" in EXPECTED_PREFLIGHT_ARMS
    for prebatch in (
        "cas-completeness",
        "disk-floor",
        "roots-contract",
        "store-mount",
    ):
        assert prebatch in EXPECTED_PREFLIGHT_ARMS, prebatch
    assert len(set(EXPECTED_PREFLIGHT_ARMS)) == len(EXPECTED_PREFLIGHT_ARMS)


# ---------------------------------------------------------------------------
# Cell 10: leg-event closed kind vocabulary (leg_generation validator)
# ---------------------------------------------------------------------------


def _leg_event_body(event: str, **extra: Any) -> dict[str, Any]:
    """One minimal leg-event body of the given kind, self-digest stamped."""

    body: dict[str, Any] = {
        "schema_version": LEG_EVENT_SCHEMA_VERSION,
        "event": event,
        "campaign_id": "campaign-ab30",
        "generation_id": "generation-ab30",
        "run_id": "run-ab30",
        "created_at": "2026-08-24T00:00:00Z",
        **extra,
    }
    return stamp_row_digest(body)


def test_leg_event_vocabulary_gains_leg_started_and_leg_ended() -> None:
    """Cell 10a: the W6 lifecycle rows parse through the REAL row validator."""

    started = _leg_event_body(
        "leg_started",
        invocation_origin="supervised-restart",
        invocation_ordinal=1,
        pid=4242,
    )
    outcome = classify_reader(lambda: validate_leg_event_row(started))
    assert_era_outcome(
        reader_id="validate-leg-started-row",
        outcome=outcome,
        prebatch_expected="parsed",
    )
    ended = _leg_event_body(
        "leg_ended",
        leg_started_sha256=str(started["row_sha256"]),
        exit_class="close_minted",
        cause_fingerprint="cause-ab30",
        terminal_delta=0,
        invocation_origin="supervised-restart",
        detail="",
        synthesized=False,
    )
    outcome = classify_reader(lambda: validate_leg_event_row(ended))
    assert_era_outcome(
        reader_id="validate-leg-ended-row",
        outcome=outcome,
        prebatch_expected="parsed",
    )
    # Era honesty: a pre-batch kind still parses under the widened vocabulary.
    prior = _leg_event_body("leg_run")
    outcome = classify_reader(lambda: validate_leg_event_row(prior))
    assert_era_outcome(
        reader_id="validate-prebatch-leg-run-row",
        outcome=outcome,
        prebatch_expected="parsed",
    )


def test_unknown_leg_event_kind_refuses_typed() -> None:
    """Cell 10b: an unreviewed leg-event kind refuses TYPED, never half-parses."""

    alien = _leg_event_body(_FUTURE_SPELLING)
    outcome = classify_reader(lambda: validate_leg_event_row(alien))
    assert_era_outcome(
        reader_id="validate-unknown-leg-event-kind",
        outcome=outcome,
        prebatch_expected="refused-typed",
    )
    assert isinstance(outcome[1], LegGenerationError)
    assert "outside the closed vocabulary" in str(outcome[1])


# ---------------------------------------------------------------------------
# Cell 11: EN-8 overlay-binding carve-out (exact enumeration + typed refusal)
# ---------------------------------------------------------------------------

_GRAPH_DIGEST = "sha256:570ae5eea1eb6fe3c605ae3a783514b3e1aeb7bfca539a027435e2e78d8edc83"
_SPEECH_DIGEST = "sha256:e0d8fcd895a00a784b2cfa8db15926993de397f48a286e708a3d97591eddbc0d"

_OVERLAY_ROWS = (
    {"name": "demo-pkg", "version": "1.0.0", "sha256": "sha256:" + "ab" * 32},
)
_EXPORT = b'{"packages": []}\n'


def _unbound_manifest() -> bytes:
    """A fresh pinned overlay manifest carrying NO conda-export binding line."""

    bound = render_pip_overlay_manifest(
        _OVERLAY_ROWS, conda_export_sha256="sha256:" + "cd" * 32
    )
    lines = [
        line
        for line in bound.decode("utf-8").splitlines()
        if not line.startswith("# conda-resolved-export-sha256:")
    ]
    content = ("\n".join(lines) + "\n").encode()
    assert pip_overlay_export_binding(content) is None
    return content


def _overlay(manifest_bytes: bytes, tmp_path: Path) -> SimpleNamespace:
    manifest_path = tmp_path / "osx-arm64.pip-overlay.txt"
    manifest_path.write_bytes(manifest_bytes)
    return SimpleNamespace(
        status="pinned",
        manifest_bytes=manifest_bytes,
        manifest_path=manifest_path,
        export_binding=pip_overlay_export_binding(manifest_bytes),
        rows=lambda: [dict(row) for row in _OVERLAY_ROWS],
    )


def _lock(export_bytes: bytes) -> SimpleNamespace:
    return SimpleNamespace(
        status="locked",
        target="osx-arm64",
        lock_bytes=b"https://conda.example/x.conda#" + b"1" * 64 + b"\n",
        export_bytes=export_bytes,
        declared_export_hash=hash_bytes(export_bytes),
    )


def test_legacy_unbound_overlay_carveout_is_exactly_the_two_digests() -> None:
    """Cell 11a: the EN-8 carve-out enumerates EXACTLY graph + speech.

    A carve-out member is a reviewed waiver arm; exact-set equality means a
    widened enumeration (a laundered fresh unbound manifest) fails here, in
    addition to the lane's own self-retiring committed-manifest assertion.
    """

    assert LEGACY_UNBOUND_OVERLAY_MANIFESTS == frozenset(
        {_GRAPH_DIGEST, _SPEECH_DIGEST}
    )


def test_fresh_unbound_overlay_refuses_typed(tmp_path: Path) -> None:
    """Cell 11b: a non-enumerated unbound manifest refuses at both real guards."""

    overlay = _overlay(_unbound_manifest(), tmp_path)
    assert hash_bytes(overlay.manifest_bytes) not in LEGACY_UNBOUND_OVERLAY_MANIFESTS
    # envs guard: an unbound overlay outside the enumeration forms NO generation.
    assert (
        compute_environment_generation(
            "demo", _lock(_EXPORT), SimpleNamespace(
                imports=("a",), export_checks=(), source_build=()
            ),
            pip_overlay=overlay,
        )
        is None
    )
    # Lifecycle guard: the install path refuses TYPED before touching pip.
    intent = SimpleNamespace(name="demo", pip_overlay=overlay, lock=_lock(_EXPORT))
    host = SimpleNamespace(_backend=SimpleNamespace(overlay=None))
    outcome = classify_reader(
        lambda: SequentialEnvironmentLifecycle._apply_pip_overlay(
            host, intent, Path("/nonexistent-prefix")
        )
    )
    assert_era_outcome(
        reader_id="lifecycle-apply-fresh-unbound-overlay",
        outcome=outcome,
        prebatch_expected="refused-typed",
    )
    assert isinstance(outcome[1], EnvironmentOverlayError)


# ---------------------------------------------------------------------------
# Negative controls: closure direction on representative widened sets
# ---------------------------------------------------------------------------


def test_negative_control_unreviewed_spelling_is_nowhere() -> None:
    """An unreviewed spelling is a member of NO widened vocabulary."""

    representative_sets: Mapping[str, Any] = {
        "OperationalEventKind": _enum_values(OperationalEventKind),
        "InvocationOrigin": _enum_values(InvocationOrigin),
        "ROTATION_CLASSES": ROTATION_CLASSES,
        "SCHEDULER_QUARANTINE_REASONS": SCHEDULER_QUARANTINE_REASONS,
        "GRANT_RELEASE_CENSUS_CLASSES": frozenset(GRANT_RELEASE_CENSUS_CLASSES),
        "MACHINE_DEFERRAL_DISPOSITIONS": MACHINE_DEFERRAL_DISPOSITIONS,
        "ATTEMPT_REFUND_LOCAL_OUTCOMES": ATTEMPT_REFUND_LOCAL_OUTCOMES,
        "EXPECTED_PREFLIGHT_ARMS": frozenset(EXPECTED_PREFLIGHT_ARMS),
        "CHECKER_LANE_FAILURE_CLASSES": CHECKER_LANE_FAILURE_CLASSES,
        "LEGACY_UNBOUND_OVERLAY_MANIFESTS": LEGACY_UNBOUND_OVERLAY_MANIFESTS,
    }
    for name, members in representative_sets.items():
        assert _FUTURE_SPELLING not in members, name
