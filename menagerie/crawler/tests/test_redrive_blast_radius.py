"""Named-mode blast-radius guard for --redrive-targets (D-54).

A named re-drive is an operator act with a real blast radius: every enumerated
member buys a fresh ``dependency-refresh`` work generation (author plus checker
sessions, certain by construction -- fa60bef4). Until this guard the target
list was a bare CSV: no reason, no cap, no binding to the intake census it was
composed against, and no disclosure of which named members selection actually
admitted. These tests pin the four guard arms:

* a re-drive REQUIRES a durable reason;
* inline target lists are CAPPED; above the cap the list must arrive as a
  self-digested manifest bound to the exact intake snapshot
  (over-cap-requires-manifest is the D-54 RED acceptance);
* the manifest refuses typed on any structural or binding violation --
  wrong field set, failed self-digest, empty/duplicate targets, or a
  snapshot sha that is not the ACTIVE intake's;
* the guard adds NO selection semantics: the continuation identity of a
  config with a reason and manifest is byte-identical to the same targets
  without them (nothing rotates).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from menagerie.crawler.driver_contracts import (
    REDRIVE_INLINE_TARGET_CAP,
    DriverConfig,
)
from menagerie.crawler.redrive_manifest import (
    REDRIVE_MANIFEST_FORMAT,
    RedriveManifestError,
    load_redrive_manifest,
    render_redrive_manifest,
)
from menagerie.crawler.work_selection import selection_continuation_identity

_TARGETS = tuple(f"m_{index:04d}" for index in range(REDRIVE_INLINE_TARGET_CAP + 1))


@pytest.mark.smoke
def test_over_cap_inline_targets_require_a_digested_manifest() -> None:
    """The D-54 RED acceptance: above the inline cap, only a manifest admits."""

    with pytest.raises(ValueError, match="inline cap"):
        DriverConfig(
            redrive_targets=_TARGETS,
            redrive_reason="operator: dependency refresh wave",
        )


@pytest.mark.smoke
def test_redrive_targets_require_a_reason() -> None:
    """A named re-drive without a durable rationale refuses typed."""

    with pytest.raises(ValueError, match="redrive-reason"):
        DriverConfig(redrive_targets=("m_0001",))
    # Whitespace is not a rationale.
    with pytest.raises(ValueError, match="redrive-reason"):
        DriverConfig(redrive_targets=("m_0001",), redrive_reason="   ")


@pytest.mark.smoke
def test_at_cap_inline_targets_with_reason_are_accepted(tmp_path: Path) -> None:
    """The cap is a cap, not a lower bound: at-cap inline lists stay legal."""

    config = DriverConfig(
        redrive_targets=_TARGETS[:REDRIVE_INLINE_TARGET_CAP],
        redrive_reason="operator: dependency refresh wave",
    )
    assert len(config.redrive_targets) == REDRIVE_INLINE_TARGET_CAP

    # Over-cap IS legal once the list arrives as a digested manifest.
    manifest_path = tmp_path / "redrive.json"
    manifest_path.write_text(
        json.dumps(
            render_redrive_manifest(
                reason="operator: dependency refresh wave",
                intake_snapshot_sha256="sha256:feed",
                targets=_TARGETS,
            )
        ),
        encoding="utf-8",
    )
    config = DriverConfig(
        redrive_targets=_TARGETS,
        redrive_reason="operator: dependency refresh wave",
        redrive_manifest_path=manifest_path.resolve(),
    )
    assert len(config.redrive_targets) == REDRIVE_INLINE_TARGET_CAP + 1


@pytest.mark.smoke
def test_manifest_path_contract() -> None:
    """A manifest path must be absolute and must accompany actual targets."""

    with pytest.raises(ValueError, match="absolute"):
        DriverConfig(
            redrive_targets=("m_0001",),
            redrive_reason="operator: refresh",
            redrive_manifest_path=Path("relative/manifest.json"),
        )
    with pytest.raises(ValueError, match="redrive_manifest_path"):
        DriverConfig(redrive_manifest_path=Path("/tmp/manifest.json"))


@pytest.mark.smoke
def test_manifest_round_trips_and_refuses_typed(tmp_path: Path) -> None:
    """parse(render(m)) == m; every structural violation refuses typed."""

    body = render_redrive_manifest(
        reason="operator: refresh",
        intake_snapshot_sha256="sha256:feed",
        targets=("m_0001", "m_0002"),
    )
    path = tmp_path / "redrive.json"
    path.write_text(json.dumps(body), encoding="utf-8")
    manifest = load_redrive_manifest(path)
    assert manifest.reason == "operator: refresh"
    assert manifest.intake_snapshot_sha256 == "sha256:feed"
    assert manifest.targets == ("m_0001", "m_0002")
    assert manifest.manifest_sha256 == body["manifest_sha256"]
    # Snapshot binding: the ACTIVE intake or nothing.
    manifest.require_snapshot("sha256:feed")
    with pytest.raises(RedriveManifestError, match="never carried forward by hand"):
        manifest.require_snapshot("sha256:other")

    # Tampered content fails its own digest.
    tampered = dict(body)
    tampered["targets"] = ["m_0001", "m_0003"]
    path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(RedriveManifestError, match="content digest"):
        load_redrive_manifest(path)

    # Wrong format tag refuses (no downgrade ladder).
    wrong_format = dict(body)
    wrong_format["format"] = "menagerie.crawler.redrive-manifest.v0"
    path.write_text(json.dumps(wrong_format), encoding="utf-8")
    with pytest.raises(RedriveManifestError, match="format"):
        load_redrive_manifest(path)

    # Field-set drift refuses (exact contract, subset and superset alike).
    extra = dict(body)
    extra["operator"] = "jmt"
    path.write_text(json.dumps(extra), encoding="utf-8")
    with pytest.raises(RedriveManifestError, match="field contract"):
        load_redrive_manifest(path)

    # Duplicate targets refuse.
    duplicated = render_redrive_manifest(
        reason="operator: refresh",
        intake_snapshot_sha256="sha256:feed",
        targets=("m_0001", "m_0001"),
    )
    path.write_text(json.dumps(duplicated), encoding="utf-8")
    with pytest.raises(RedriveManifestError, match="unique"):
        load_redrive_manifest(path)

    assert REDRIVE_MANIFEST_FORMAT == "menagerie.crawler.redrive-manifest.v1"


@pytest.mark.smoke
def test_wake_argv_replays_the_manifest_not_the_expanded_inline_list(
    tmp_path: Path,
) -> None:
    """A wake fired on a manifest launch must resume THROUGH the manifest.

    Replaying the expanded inline CSV would refuse at the resumed leg's own
    over-cap guard -- the exact hand-enumerated-flag-list class D1 adjudicated.
    An inline launch replays its targets AND its reason.
    """

    from dataclasses import replace

    from menagerie.crawler.tests.test_slice_f_driver import _driver, _snapshot

    manifest_path = tmp_path / "redrive.json"
    manifest_path.write_text(
        json.dumps(
            render_redrive_manifest(
                reason="operator: refresh",
                intake_snapshot_sha256="sha256:feed",
                targets=_TARGETS,
            )
        ),
        encoding="utf-8",
    )
    snapshot = _snapshot(tmp_path, count=1)
    driver = _driver(tmp_path, snapshot)
    driver.config = replace(
        driver.config,
        redrive_targets=_TARGETS,
        redrive_reason="operator: refresh",
        redrive_manifest_path=manifest_path.resolve(),
    )
    argv = driver._wakeup_callback_argv()  # noqa: SLF001
    assert "--redrive-manifest" in argv
    assert argv[argv.index("--redrive-manifest") + 1] == str(manifest_path.resolve())
    assert "--redrive-targets" not in argv
    assert "--redrive-reason" not in argv

    driver.config = replace(
        driver.config,
        redrive_targets=("m_0001", "m_0002"),
        redrive_reason="operator: two-member surgical refresh",
        redrive_manifest_path=None,
    )
    argv = driver._wakeup_callback_argv()  # noqa: SLF001
    assert "--redrive-manifest" not in argv
    assert argv[argv.index("--redrive-targets") + 1] == "m_0001,m_0002"
    assert argv[argv.index("--redrive-reason") + 1] == (
        "operator: two-member surgical refresh"
    )


@pytest.mark.slow
def test_redrive_leg_discloses_the_admitted_delta_before_minting_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A named re-drive leg durably discloses named vs admitted vs excluded.

    One real failed member is named beside one phantom ID that selection can
    never admit: the preview must count both truthfully -- a silently dropped
    named target is exactly how a mis-composed re-drive wave hides.
    """

    from dataclasses import replace

    from menagerie.crawler.recordio import scan_jsonl
    from menagerie.crawler.tests.test_slice_f_driver import (
        _BLOCKED_SOURCE_PAYLOAD,
        _driver,
        _install_executor_lane,
        _paths,
        _snapshot,
    )
    from menagerie.crawler.driver import CommandAuthorLane

    snapshot = _snapshot(tmp_path, count=1)
    stable_id = snapshot.items[0].stable_id
    paths = _paths(tmp_path, snapshot)
    _install_executor_lane(monkeypatch, _BLOCKED_SOURCE_PAYLOAD)
    assert (
        _driver(tmp_path, snapshot, author=CommandAuthorLane(("fake-author",))).run().status
        == "complete"
    )

    redrive = _driver(tmp_path, snapshot, author=CommandAuthorLane(("fake-author",)))
    redrive.config = replace(
        redrive.config,
        redrive_targets=(stable_id, "m_phantom_never_intake"),
        redrive_reason="operator: dependency refresh probe",
    )
    redrive.run()

    deltas = [
        event
        for event in scan_jsonl(paths.operational_ledger)
        if event.get("details", {}).get("disposition") == "redrive-admitted-delta"
    ]
    assert len(deltas) == 1
    details = deltas[0]["details"]
    assert details["named"] == 2
    assert details["admitted"] == 1
    assert details["not_admitted"] == 1
    assert details["not_admitted_sample"] == ["m_phantom_never_intake"]
    assert details["redrive_reason"] == "operator: dependency refresh probe"
    assert details["redrive_manifest_sha256"] is None


@pytest.mark.smoke
def test_guard_fields_do_not_rotate_the_continuation_identity(tmp_path: Path) -> None:
    """Reason and manifest are audit surface, never selection semantics.

    The wake continuation identity seals what a resumed leg SELECTS; the
    guard's fields change only what is disclosed about the same selection, so
    the identity must be byte-identical with and without them -- otherwise
    D-54 would rotate an identity axis and belong to the Wave-2 batch by law.
    """

    targets = ("m_0001", "m_0002")
    manifest_path = tmp_path / "redrive.json"
    manifest_path.write_text(
        json.dumps(
            render_redrive_manifest(
                reason="operator: refresh",
                intake_snapshot_sha256="sha256:feed",
                targets=targets,
            )
        ),
        encoding="utf-8",
    )
    with_guard = DriverConfig(
        redrive_targets=targets,
        redrive_reason="operator: refresh",
        redrive_manifest_path=manifest_path.resolve(),
    )
    bare = DriverConfig(
        redrive_targets=targets,
        redrive_reason="operator: a different rationale entirely",
    )
    assert selection_continuation_identity(with_guard) == (
        selection_continuation_identity(bare)
    )
