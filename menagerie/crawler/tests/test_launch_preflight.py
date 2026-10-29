"""Launch-preflight tests: a promised environment rotation must actually rotate.

Rung-12 census, top fix 1: two long legs launched on the belief that wave-9
``environment.yml`` edits had rotated the quarantined core environment, but the
park identity hashes the four COMMITTED setup artifacts, none of which changed,
so all ~43 rebuild parks were re-claimed before env/worker/reducer. These tests
pin the tripwire that refuses such a launch.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from menagerie.crawler.driver_admission import (
    _intent_setup_artifact_identity,
    _intent_setup_artifact_state,
)
from menagerie.crawler.environment_rebuild import (
    REBUILD_DEFERRAL_RELATIVE_PATH,
    append_rebuild_deferral_row,
    build_rebuild_deferral_row,
)
from menagerie.crawler.envs import DEFAULT_ENVS_ROOT, load_environment_registry
from menagerie.crawler.operator_environment import resolve_environment_state_root
from menagerie.crawler.tools.launch_preflight import (
    DISK_FLOOR_UNMET,
    MIXED_FAMILY,
    PRODUCTION_ENVIRONMENT_DISK_FLOOR_BYTES,
    ROTATION_DID_NOT_ROTATE,
    STATE_ROOT_UNBOUND,
    UNSTABLE_PROBE_RECEIPT,
    evaluate_disk_floor,
    evaluate_operator_state_root,
    evaluate_probe_receipt_stability,
    evaluate_rebuild_parks,
    main,
)

#: Sub-second load-bearing coverage promoted into the mandatory smoke tier (T5-7).
pytestmark = pytest.mark.smoke

#: Intents deliberately kept lockless (routed away from model work); the
#: lockless-family assert requires naming them explicitly at launch. ``mmlab``
#: and ``oddballs`` graduated to locked families in the 2026-08-19 rotations,
#: so allowing them here would silently suppress their integrity validation.
_LOCKLESS_ALLOWANCES = (
    "--allow-lockless",
    "tf-keras-arm64",
    "--allow-lockless",
    "jax-flax-arm64",
)

#: Shipped osx-arm64 families whose committed probe receipts carry rotation-host
#: text on PASSED probes, so the lane's post-probe re-attestation must rewrite a
#: provenance-bound member (see ``evaluate_probe_receipt_stability``). These
#: fixtures copy the live registry, so they name the same allowance a launch on
#: today's committed families has to name. Every entry here is an intent awaiting
#: a rotation; shrink the tuple as they rotate, never grow it to quiet a failure.
_UNSTABLE_RECEIPT_ALLOWANCES = tuple(
    argument
    for intent in ("graph", "audio", "speech", "recsys", "detect-misc", "paddle-arm64")
    for argument in ("--allow-unstable-probe-receipt", intent)
)

#: Fixture parameterization (W2 calibration-mode arm): these trivial fixture
#: campaigns adjudicate nothing, so they DECLARE the F-O19 vacuous case
#: explicitly instead of holding a receipt. The arm's own XOR semantics are
#: exercised in ``test_checker_calibration_mode_arm.py``; never add this flag
#: to a fixture that models a verdict-rendering launch.
_NO_VERDICT_DECLARATION = ("--renders-no-gate-verdicts",)

#: Every named allowance a launch against the live committed registry must pass.
_ALLOWANCES = (
    *_LOCKLESS_ALLOWANCES,
    *_UNSTABLE_RECEIPT_ALLOWANCES,
    *_NO_VERDICT_DECLARATION,
)


def _campaign(tmp_path: Path, *, intent: str = "core") -> tuple[Path, Path, str]:
    """Build one campaign fixture whose park row matches the live registry.

    Returns
    -------
    tuple[pathlib.Path, pathlib.Path, str]
        Records root, envs root, and the parked quarantined identity.
    """

    envs_root = tmp_path / "envs"
    shutil.copytree(DEFAULT_ENVS_ROOT, envs_root)
    registry = load_environment_registry(envs_root)
    identity = _intent_setup_artifact_identity(registry.intents[intent])
    state = _intent_setup_artifact_state(registry.intents[intent])
    records_root = tmp_path / "records"
    (records_root / "models").mkdir(parents=True)
    row = build_rebuild_deferral_row(
        stable_id="m_parked",
        work_id="work-m_parked",
        name="Parked",
        campaign_id="campaign-preflight",
        run_id="run-1",
        machine_id="machine-1",
        created_at="2026-08-09T00:00:00Z",
        quarantine_event_id="environment-cleanup-preflight1",
        intent=intent,
        target="osx-arm64",
        artifact_identity=identity,
        artifact_state=state,
        env_generation="sha256:" + "22" * 32,
        failure_type="menagerie.crawler.env_lifecycle.DiskRecoveryError",
    )
    append_rebuild_deferral_row(records_root / REBUILD_DEFERRAL_RELATIVE_PATH, row)
    return records_root, envs_root, identity


def test_preflight_refuses_when_the_promised_rotation_did_not_rotate(
    tmp_path: Path, capsys
) -> None:
    """An unchanged committed setup identity holds the park and fails preflight.

    This is exactly the rung-12 state: park rows live, setup artifacts
    byte-identical to the quarantined family, launch about to waste a leg.
    """

    records_root, envs_root, identity = _campaign(tmp_path)
    report = evaluate_rebuild_parks(
        records_root=records_root, envs_root=envs_root, target="osx-arm64"
    )
    assert report["rebuild_rows"] == 1
    assert report["still_held"] == 1
    assert report["released"] == 0
    core = next(item for item in report["intents"] if item["intent"] == "core")
    assert core["current_identity"] == identity
    assert core["quarantined_identities"] == [identity]

    code = main(
        [
            "--records-root",
            str(records_root),
            "--envs-root",
            str(envs_root),
            *_ALLOWANCES,
        ]
    )
    captured = capsys.readouterr()
    assert code == 2
    assert ROTATION_DID_NOT_ROTATE in captured.err
    assert "environment.yml edit is not a rotation" in captured.err
    # Report-only downgrades the refusal for launches promising no rotation.
    assert (
        main(
            [
                "--records-root",
                str(records_root),
                "--envs-root",
                str(envs_root),
                *_ALLOWANCES,
                "--report-only",
            ]
        )
        == 0
    )


def _publish_rotated_core_family(envs_root: Path) -> None:
    """Rotate the copied core family through the REAL publication path.

    The edit-spec-then-rotate flow: the spec changes first, then a consistent
    family (members + provenance binding the edited spec) is published through
    ``publish_rotation_family`` -- exactly what a completed rotation leaves.
    """

    import json

    from menagerie.crawler.env_lifecycle import expected_probe_names
    from menagerie.crawler.tests.rotation_transaction_support import (
        family_members,
        rotation_provenance,
        solve_fixture,
    )
    from menagerie.crawler.tools.rotate_setup_artifacts import publish_rotation_family

    (envs_root / "core" / "environment.yml").write_text(
        json.dumps({"name": "core", "channels": ["conda-forge"], "dependencies": ["python"]}),
        encoding="utf-8",
    )
    registry = load_environment_registry(envs_root)
    fixture = solve_fixture(
        "rotated", probe_names=expected_probe_names(registry.intents["core"].probes)
    )
    members = family_members(fixture, rotation_provenance(envs_root, "core", fixture))
    publish_rotation_family(envs_root / "core" / "locks", "osx-arm64", members)


def test_preflight_passes_once_the_setup_artifacts_actually_rotated(
    tmp_path: Path,
) -> None:
    """A completed rotation releases the park and the preflight passes."""

    records_root, envs_root, identity = _campaign(tmp_path)
    _publish_rotated_core_family(envs_root)

    report = evaluate_rebuild_parks(
        records_root=records_root, envs_root=envs_root, target="osx-arm64"
    )
    assert report["rebuild_rows"] == 1
    assert report["still_held"] == 0
    assert report["released"] == 1
    core = next(item for item in report["intents"] if item["intent"] == "core")
    assert core["current_identity"] != identity
    assert (
        main(
            [
                "--records-root",
                str(records_root),
                "--envs-root",
                str(envs_root),
                # Host-independent fixture parameterization (the
                # _green_arguments spelling): production keeps the defaults;
                # the floor's own enforcement has dedicated tests.
                "--disk-floor-bytes",
                str(1024**3),
                "--allocation-probe-bytes",
                str(1024**2),
                "--placement-anchor",
                str(tmp_path),
                *_ALLOWANCES,
            ]
        )
        == 0
    )


def test_preflight_refuses_a_hand_mutated_member_as_mixed_family(tmp_path: Path, capsys) -> None:
    """A member-byte change WITHOUT its rotation refuses; it never releases.

    This is the T3-T7 hole inverted: the park-release claimant reacts to any
    byte change, so before the family validation existed this exact state --
    an appended byte on the committed lock -- read as a completed rotation
    and passed the preflight.
    """

    records_root, envs_root, _identity = _campaign(tmp_path)
    lock_path = envs_root / "core" / "locks" / "osx-arm64.lock"
    assert lock_path.is_file(), "fixture expects the committed rotated core family"
    lock_path.write_bytes(lock_path.read_bytes() + b"# rotated\n")

    code = main(
        [
            "--records-root",
            str(records_root),
            "--envs-root",
            str(envs_root),
            *_ALLOWANCES,
        ]
    )
    captured = capsys.readouterr()
    assert code == 2
    assert MIXED_FAMILY in captured.err


def test_asserted_state_root_must_be_the_one_the_operator_will_resolve(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A CAS receipt proves nothing about a root the launch will not use.

    X4: the CAS is seeded at an explicit external state root, but the shipped
    operator resolves its own root from ``MENAGERIE_ENVIRONMENT_STATE_ROOT``
    else the sole repository-anchored resolver. Nothing bound the
    ``--state-root`` the preflight verifies to the root the operator will
    actually open, so a launch that forgets to export the variable passes a
    green CAS receipt and then builds offline against a different, incomplete
    CAS.
    """

    seeded = tmp_path / "seeded-state"
    seeded.mkdir()
    launch_cwd = tmp_path / "campaign"
    launch_cwd.mkdir()

    monkeypatch.delenv("MENAGERIE_ENVIRONMENT_STATE_ROOT", raising=False)
    report = evaluate_operator_state_root(state_root=seeded, launch_cwd=launch_cwd)
    assert report["ok"] is False
    assert report["environment_variable_set"] is False
    assert report["resolved"] == str(resolve_environment_state_root(launch_cwd))
    assert STATE_ROOT_UNBOUND in str(report["error"])

    # Exported and equal: the receipt now describes the root that will be used.
    monkeypatch.setenv("MENAGERIE_ENVIRONMENT_STATE_ROOT", str(seeded))
    bound = evaluate_operator_state_root(state_root=seeded, launch_cwd=launch_cwd)
    assert bound["ok"] is True
    assert bound["environment_variable_set"] is True

    # Exported but pointing elsewhere is the same defect, loudly.
    other = tmp_path / "other-state"
    other.mkdir()
    monkeypatch.setenv("MENAGERIE_ENVIRONMENT_STATE_ROOT", str(other))
    mismatched = evaluate_operator_state_root(state_root=seeded, launch_cwd=launch_cwd)
    assert mismatched["ok"] is False
    assert mismatched["resolved"] == str(other)
    assert STATE_ROOT_UNBOUND in str(mismatched["error"])

    # A relative export is refused exactly as the operator refuses it.
    monkeypatch.setenv("MENAGERIE_ENVIRONMENT_STATE_ROOT", "relative/state")
    relative = evaluate_operator_state_root(state_root=seeded, launch_cwd=launch_cwd)
    assert relative["ok"] is False
    assert "must be absolute" in str(relative["error"])


def test_state_root_binding_refuses_a_launch_end_to_end(
    tmp_path: Path, capsys, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The unbound state root refuses the whole preflight, typed."""

    records_root, envs_root, _identity = _campaign(tmp_path)
    _publish_rotated_core_family(envs_root)
    seeded = tmp_path / "seeded-state"
    seeded.mkdir()
    monkeypatch.delenv("MENAGERIE_ENVIRONMENT_STATE_ROOT", raising=False)

    code = main(
        [
            "--records-root",
            str(records_root),
            "--envs-root",
            str(envs_root),
            "--state-root",
            str(seeded),
            *_ALLOWANCES,
        ]
    )
    captured = capsys.readouterr()
    assert code == 2
    assert STATE_ROOT_UNBOUND in captured.err
    assert "MENAGERIE_ENVIRONMENT_STATE_ROOT" in captured.err


def test_disk_floor_is_asserted_against_every_lane_write_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A store that already sits under the lane's floor refuses before launch.

    X4: ``evaluate_store_mount`` proves the store is mounted and measurable but
    never compares the measurement against the floor the production lane
    enforces (``build_command_environment_lane`` -> 30 GiB), so a launch on a
    starved host passes preflight and then parks every intent behind the
    disk-floor valve one build at a time.
    """

    env_root = tmp_path / "envs"
    env_root.mkdir()
    state_root = tmp_path / "state"
    state_root.mkdir()

    starved = {"free": PRODUCTION_ENVIRONMENT_DISK_FLOOR_BYTES - 1}

    def _fake_free(path: Path) -> int:
        """Report the starved measurement for every probed destination."""

        return int(starved["free"])

    report = evaluate_disk_floor(env_root=env_root, state_root=state_root, disk_free=_fake_free)
    assert report["ok"] is False
    assert report["minimum_free_bytes"] == PRODUCTION_ENVIRONMENT_DISK_FLOOR_BYTES
    assert DISK_FLOOR_UNMET in str(report["error"])
    roles = {row["role"] for row in report["destinations"]}
    assert {"env-root", "backend-state", "tmp"} <= roles

    starved["free"] = PRODUCTION_ENVIRONMENT_DISK_FLOOR_BYTES
    assert (
        evaluate_disk_floor(env_root=env_root, state_root=state_root, disk_free=_fake_free)["ok"]
        is True
    )


def test_disk_floor_uses_explicit_launch_tmp_root_as_its_authority(tmp_path: Path) -> None:
    """Preflight measures the same explicit temp filesystem the lane will use."""

    env_root = tmp_path / "envs"
    state_root = tmp_path / "state"
    tmp_root = tmp_path / "launch-tmp"
    for path in (env_root, state_root, tmp_root):
        path.mkdir()

    def _fake_free(path: Path) -> int:
        """Make only the explicitly configured launch temp root starved."""

        return (
            PRODUCTION_ENVIRONMENT_DISK_FLOOR_BYTES - 1
            if path == tmp_root
            else PRODUCTION_ENVIRONMENT_DISK_FLOOR_BYTES * 2
        )

    report = evaluate_disk_floor(
        env_root=env_root,
        state_root=state_root,
        tmp_root=tmp_root,
        disk_free=_fake_free,
    )
    assert report["ok"] is False
    assert report["authority"] == {
        "role": "tmp",
        "destination": str(tmp_root),
        "probe": str(tmp_root),
        "free_bytes": PRODUCTION_ENVIRONMENT_DISK_FLOOR_BYTES - 1,
    }
    assert "tmp" in str(report["error"])


def test_a_committed_receipt_that_a_launch_must_re_attest_is_refused_up_front(
    tmp_path: Path, capsys
) -> None:
    """Host-dependent passed-probe detail is caught BEFORE the leg, not after it.

    X4, observed for real: the lane re-attests ``<target>.probes.json`` after
    probes pass, and that member is provenance-bound. When a committed receipt's
    PASSED rows carry host- or path-dependent chatter (the shipped audio /
    speech / detect-misc / paddle receipts literally embed the rotation host's
    absolute ``/opt/crawl/.crawler-envs/striker-e-rotation/envs/...`` prefixes,
    which no launch prefix can ever equal), the re-attestation necessarily
    rewrites the member -- and then ``evaluate_family_integrity`` refuses the
    NEXT launch with ``mixed-family``. Discovering that after burning a leg is
    the exact serial-blocker pattern this preflight exists to kill, so the
    unstable receipt is a refusal in its own right, with the same explicit
    named-allowance escape as the lockless intents.
    """

    import json

    from menagerie.crawler.env_lifecycle import (
        ProbeResult,
        canonical_probe_receipt_bytes,
    )

    records_root, envs_root, _identity = _campaign(tmp_path)

    # core ships a canonical receipt today, so the census must clear it.
    stable = evaluate_probe_receipt_stability(envs_root=envs_root, target="osx-arm64")
    assert "core" not in {row["intent"] for row in stable["unstable"]}
    assert "core" in {row["intent"] for row in stable["intents"] if row["checked"]}

    # Republish core's receipt exactly as a rotation on a noisy host would:
    # canonical bytes, every probe passed, rotation-host text in one detail.
    receipt_path = (
        load_environment_registry(envs_root)
        .intents["core"]
        .lock.lock_path.with_name("osx-arm64.probes.json")
    )
    payload = json.loads(receipt_path.read_bytes())
    assert all(row["passed"] for row in payload["probes"])
    noisy_name = str(payload["probes"][0]["name"])
    receipt_path.write_bytes(
        canonical_probe_receipt_bytes(
            [
                ProbeResult(
                    str(row["name"]),
                    bool(row["passed"]),
                    (
                        "OMP: Warning #179: /opt/crawl/.crawler-envs/"
                        "striker-e-rotation/envs/core-1c724"
                        if str(row["name"]) == noisy_name
                        else str(row["detail"])
                    ),
                )
                for row in payload["probes"]
            ]
        )
    )

    report = evaluate_probe_receipt_stability(envs_root=envs_root, target="osx-arm64")
    assert report["ok"] is False
    unstable = {row["intent"]: row for row in report["unstable"]}
    assert "core" in unstable, report["intents"]
    assert unstable["core"]["rows"][0]["name"] == noisy_name
    assert "OMP" in unstable["core"]["rows"][0]["detail"]

    # The named allowance is the only escape, and it is disposition-negative:
    # it lets the launch proceed while naming the de-binding in the launch log.
    allowed = evaluate_probe_receipt_stability(
        envs_root=envs_root,
        target="osx-arm64",
        allow_unstable=(
            "core",
            "graph",
            "audio",
            "speech",
            "recsys",
            "detect-misc",
            "paddle-arm64",
        ),
    )
    assert allowed["ok"] is True
    assert "core" in allowed["allowed_unstable"]

    # End to end: the unstable receipts alone trip the typed refusal, and the
    # named allowances are the only thing that clears that refusal (the live
    # rebuild park keeps refusing for its own, separate reason).
    arguments = [
        "--records-root",
        str(records_root),
        "--envs-root",
        str(envs_root),
        *_LOCKLESS_ALLOWANCES,
    ]
    assert main(arguments) == 2
    assert UNSTABLE_PROBE_RECEIPT in capsys.readouterr().err

    assert (
        main([*arguments, *_UNSTABLE_RECEIPT_ALLOWANCES, "--allow-unstable-probe-receipt", "core"])
        == 2
    )
    cleared = capsys.readouterr().err
    assert UNSTABLE_PROBE_RECEIPT not in cleared
    # And what remains is exactly the consequence the census predicts a launch
    # would otherwise discover only on the leg AFTER the build: republishing the
    # receipt without its rotation de-binds the provenance.
    assert MIXED_FAMILY in cleared


def test_the_shipped_committed_receipts_are_all_stable() -> None:
    """Every shipped committed receipt survives the stability census.

    The shipped osx-arm64 receipts USED to embed rotation-host text on passed
    rows (absolute prefixes, OMP tmp warnings), which the census names as
    guaranteed re-attestations. Those committed members were normalized to the
    canonical passed detail and their provenance rebound in the same change
    that canonicalized ``operator_environment._probe_result``, so the census
    over the live registry must now hold clean -- while still CHECKING every
    locked family (a census that skipped them all would be the tripwire
    disarmed by accident).
    """

    report = evaluate_probe_receipt_stability(envs_root=DEFAULT_ENVS_ROOT, target="osx-arm64")
    checked = {row["intent"] for row in report["intents"] if row.get("checked")}
    assert {"audio", "detect-misc", "paddle-arm64", "speech"} <= checked
    assert report["ok"], report["unstable"]
    assert report["unstable"] == []


def test_preflight_with_no_park_rows_passes_trivially(tmp_path: Path) -> None:
    """An empty or absent rebuild ledger never blocks a launch."""

    envs_root = tmp_path / "envs"
    shutil.copytree(DEFAULT_ENVS_ROOT, envs_root)
    records_root = tmp_path / "records"
    (records_root / "models").mkdir(parents=True)
    report = evaluate_rebuild_parks(
        records_root=records_root, envs_root=envs_root, target="osx-arm64"
    )
    assert report["rebuild_rows"] == 0
    assert report["still_held"] == 0
    assert (
        main(
            [
                "--records-root",
                str(records_root),
                "--envs-root",
                str(envs_root),
                # Host-independent fixture parameterization (the
                # _green_arguments spelling): production keeps the defaults;
                # the floor's own enforcement has dedicated tests.
                "--disk-floor-bytes",
                str(1024**3),
                "--allocation-probe-bytes",
                str(1024**2),
                "--placement-anchor",
                str(tmp_path),
                *_ALLOWANCES,
            ]
        )
        == 0
    )
