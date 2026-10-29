"""Rotation publication is a transaction: old family XOR proven new family.

C-ROTATION-TXN (T3-T7 3-lab + T5-11 2-lab + T3-T6 residual) -- the HARD gate
before ANY next rotation. The defect family these tests pin:

* ``rotate_intent``'s publication was N separate member writes: a crash
  mid-publication left a MIXED committed family whose accidental new identity
  released every rebuild park against an environment no clean create ever
  proved (T3-T7);
* the launch preflight released parks on ANY member-byte change without
  validating the family or its provenance binding, so the mixed family --
  or a hand-edited member -- read as a completed rotation (T3-T7);
* nothing at runtime compared ``spec_sha256``, so a post-rotation
  ``environment.yml`` edit kept reading as current (T3-T6 residual), and a
  reused rotation state root could serve a stale solve cache (T3-T6);
* ``rotate_intent``/``main`` had zero dynamic tests (T5-11).

The fix: publication stages the complete family in an immutable per-generation
directory, then commits it with ONE atomic manifest rename; the registry
loader reads the manifest-named generation as the family authority and fails
closed on any inconsistency; the launch preflight validates every locked
family against its committed provenance (per-member digests, spec currency,
per-intent probe currency) and refuses a mixed family loudly.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import pytest

from menagerie.crawler.driver_admission import (
    _intent_setup_artifact_identity,
)
from menagerie.crawler.environment_rebuild import (
    REBUILD_DEFERRAL_RELATIVE_PATH,
    append_rebuild_deferral_row,
    build_rebuild_deferral_row,
)
from menagerie.crawler.envs import (
    family_manifest_path,
    load_environment_registry,
    publication_journal_path,
)
from menagerie.crawler.identity import hash_bytes
from menagerie.crawler.tests.rotation_transaction_support import (
    INTENT,
    NEW_FAMILY_CUTS,
    OLD_FAMILY_CUTS,
    TARGET,
    ScriptedRotationBackend,
    family_members,
    mini_envs_root,
    populate_solve_commit,
    rotation_provenance,
    setup_artifact_identity,
    solve_fixture,
)
from menagerie.crawler.tools.launch_preflight import (
    MATERIALIZATION_PENDING,
    MIXED_FAMILY,
    STALE_PROBE_RECEIPT,
    STALE_SPEC,
    evaluate_family_integrity,
    evaluate_rebuild_parks,
    main as preflight_main,
)
from menagerie.crawler.tools.rotate_setup_artifacts import (
    RotationError,
    complete_family_publication,
    main as rotate_main,
    publish_rotation_family,
    rotate_intent,
)

#: Sub-4s load-bearing recovery coverage: the rotation transaction is the ONLY
#: writer of committed setup families, so its crash matrix belongs in the
#: mandatory per-step smoke gate.
pytestmark = pytest.mark.smoke

_OLD = solve_fixture("old")
_NEW = solve_fixture("new")
_OLD_IDENTITY = setup_artifact_identity(_OLD)
_NEW_IDENTITY = setup_artifact_identity(_NEW)
_REPO_ROOT = Path(__file__).resolve().parents[3]


def _roots(tmp_path: Path) -> SimpleNamespace:
    """Build the standard fixture universe around one mini envs root."""

    envs_root = mini_envs_root(tmp_path, old=_OLD)
    state_root = tmp_path / "rotation-state"
    state_root.mkdir()
    env_root = tmp_path / "env-store"
    env_root.mkdir()
    records_root = tmp_path / "records"
    (records_root / "models").mkdir(parents=True)
    return SimpleNamespace(envs=envs_root, state=state_root, env=env_root, records=records_root)


def _park_row(roots: SimpleNamespace) -> None:
    """Append one rebuild park row bound to the OLD committed identity."""

    from menagerie.crawler.driver_admission import _intent_setup_artifact_state

    registry = load_environment_registry(roots.envs, target=TARGET)
    intent = registry.intents[INTENT]
    row = build_rebuild_deferral_row(
        stable_id="m_parked",
        work_id="work-m_parked",
        name="Parked",
        campaign_id="campaign-rotation-txn",
        run_id="run-1",
        machine_id="machine-1",
        created_at="2026-08-11T00:00:00Z",
        quarantine_event_id="environment-cleanup-rotation-txn",
        intent=INTENT,
        target=TARGET,
        artifact_identity=_intent_setup_artifact_identity(intent),
        artifact_state=_intent_setup_artifact_state(intent),
        env_generation="sha256:" + "22" * 32,
        failure_type="menagerie.crawler.env_lifecycle.DiskRecoveryError",
    )
    append_rebuild_deferral_row(roots.records / REBUILD_DEFERRAL_RELATIVE_PATH, row)


def _current(roots: SimpleNamespace) -> SimpleNamespace:
    """Load the demo intent and derive its family disposition."""

    registry = load_environment_registry(roots.envs, target=TARGET)
    intent = registry.intents[INTENT]
    return SimpleNamespace(
        intent=intent,
        status=intent.lock.status,
        identity=_intent_setup_artifact_identity(intent),
    )


def _flat_bytes(roots: SimpleNamespace) -> dict[str, bytes]:
    """Read the FLAT committed member bytes (never the generation authority)."""

    lock_dir = roots.envs / INTENT / "locks"
    return {
        name: (lock_dir / name).read_bytes()
        for name in sorted(path.name for path in lock_dir.iterdir() if path.is_file())
        if not name.startswith(".")
    }


def _run_child(roots: SimpleNamespace, kill_at: Optional[str], *extra: str):
    """Run one rotation attempt in a child process, optionally SIGKILLed."""

    argv = [
        sys.executable,
        "-m",
        "menagerie.crawler.tests.rotation_subprocess_driver",
        "--envs-root",
        str(roots.envs),
        "--state-root",
        str(roots.state),
        "--env-root",
        str(roots.env),
        *extra,
    ]
    if kill_at is not None:
        argv.extend(["--kill-at", kill_at])
    environment = {**os.environ, "PYTHONPATH": str(_REPO_ROOT)}
    return subprocess.run(argv, capture_output=True, text=True, env=environment, timeout=120)


# --- (1) all-or-nothing at every publication cut (T5-11 RED, S variant) --------


@pytest.mark.parametrize("kill_at", [*OLD_FAMILY_CUTS, *NEW_FAMILY_CUTS])
def test_rotate_intent_publish_is_all_or_nothing_at_every_member(
    tmp_path: Path, kill_at: str
) -> None:
    """SIGKILL at every cut leaves complete-old XOR complete-proven-new.

    ``after-solve-commit`` is the kill between the operator's solve-commit and
    the family publication; ``after-publish`` the kill between publication and
    the campaign republication. Every intermediate cut is one member/phase
    promotion inside the publication transaction itself. No cut may release
    the old park row against a family that is not the complete, provenance-
    valid new one.
    """

    roots = _roots(tmp_path)
    _park_row(roots)
    old_flat = _flat_bytes(roots)

    completed = _run_child(roots, kill_at)
    assert completed.returncode == -signal.SIGKILL, (
        f"child must die by SIGKILL at {kill_at}, got rc={completed.returncode}\n"
        f"stdout: {completed.stdout}\nstderr: {completed.stderr}"
    )

    disposition = _current(roots)
    parks = evaluate_rebuild_parks(records_root=roots.records, envs_root=roots.envs, target=TARGET)
    assert disposition.status == "locked", (
        f"cut {kill_at}: the family must stay a complete readable family, "
        f"not {disposition.status!r}"
    )
    assert disposition.identity in (_OLD_IDENTITY, _NEW_IDENTITY), (
        f"cut {kill_at}: mixed-identity family {disposition.identity}"
    )
    if disposition.identity == _OLD_IDENTITY:
        # Complete OLD family: byte-identical, park still held.
        assert _flat_bytes(roots) == old_flat
        assert parks["still_held"] == 1 and parks["released"] == 0
    else:
        # Complete provenance-valid NEW family; the park releases against a
        # proven family only.
        integrity = evaluate_family_integrity(
            envs_root=roots.envs, target=TARGET, allow_lockless=()
        )
        blocking = [row for row in integrity["refused"] if row["reason"] != MATERIALIZATION_PENDING]
        assert not blocking, f"cut {kill_at}: {blocking}"
        assert parks["released"] == 1 and parks["still_held"] == 0
        # Roll-forward completes the flat materialization idempotently...
        report = complete_family_publication(roots.envs / INTENT / "locks", TARGET)
        assert report["completed"] is True
        integrity = evaluate_family_integrity(
            envs_root=roots.envs, target=TARGET, allow_lockless=()
        )
        assert integrity["ok"] is True, integrity["refused"]
        # ...and the flat members equal the new solve exactly.
        flat = _flat_bytes(roots)
        assert flat[f"{TARGET}.lock"] == _NEW.lock_bytes
        assert flat[f"{TARGET}.resolved.json"] == _NEW.export_bytes
        assert flat[f"{TARGET}.probes.json"] == _NEW.receipt_bytes
        assert not publication_journal_path(roots.envs / INTENT / "locks", TARGET).is_file()


def test_kill_between_publish_and_campaign_republication_fails_closed(
    tmp_path: Path,
) -> None:
    """A kill before republication can never serve pre-rotation campaign bytes.

    The campaign solve cache still holds the pre-rotation solve, but the
    committed family now differs from it, and the operator's committed-lock
    currency (T1-04) reads that as no cache at all. Re-running the documented
    recovery (the republication) reconciles the cache to the committed bytes.
    """

    roots = _roots(tmp_path)
    campaign_state = tmp_path / "campaign-state"
    populate_solve_commit(campaign_state, INTENT, _OLD)

    completed = _run_child(roots, "after-publish", "--campaign-state-root", str(campaign_state))
    assert completed.returncode == -signal.SIGKILL

    committed_lock = (roots.envs / INTENT / "locks" / f"{TARGET}.lock").read_bytes()
    cached_lock = (campaign_state / "solves" / INTENT / TARGET / "explicit.lock").read_bytes()
    # The stale cache is superseded by the committed family, so the operator's
    # cache-currency check refuses to serve it (pinned by
    # test_solve_cache_predating_a_rotation_is_never_served).
    assert cached_lock != committed_lock

    from menagerie.crawler.tools.rotate_setup_artifacts import (
        republish_campaign_solve_cache,
    )

    receipt = republish_campaign_solve_cache(
        campaign_state,
        rotation_state_root=roots.state,
        intent=INTENT,
        target=TARGET,
        lock_bytes=_NEW.lock_bytes,
        export_bytes=_NEW.export_bytes,
    )
    assert receipt["republished_artifacts"] >= 0
    assert (
        campaign_state / "solves" / INTENT / TARGET / "explicit.lock"
    ).read_bytes() == committed_lock


# --- (2) refusal preconditions (T5-11 RED, F variants) --------------------------


def test_rotate_intent_refuses_when_identity_does_not_change(tmp_path: Path) -> None:
    """An unchanged solve refuses BEFORE publication with zero side effects."""

    roots = _roots(tmp_path)
    old_flat = _flat_bytes(roots)
    backend = ScriptedRotationBackend(roots.state, _OLD)
    with pytest.raises(RotationError, match="did not rotate"):
        rotate_intent(
            INTENT,
            envs_root=roots.envs,
            target=TARGET,
            environment_command=("scripted-test-operator",),
            state_root=roots.state,
            env_root=roots.env,
            minimum_free_bytes=0,
            backend=backend,
            execution_lock_path=roots.state / ".test-execution.flock",
        )
    # The committed family is untouched: no member rewrite, no manifest, no
    # generation residue inside the lock directory.
    assert _flat_bytes(roots) == old_flat
    lock_dir = roots.envs / INTENT / "locks"
    assert not family_manifest_path(lock_dir, TARGET).is_file()
    assert not (lock_dir / "generations").exists()
    assert not publication_journal_path(lock_dir, TARGET).is_file()


def test_rotate_intent_publishes_only_after_probe_proof(tmp_path: Path) -> None:
    """A failed probe leaves the committed family byte-identical."""

    roots = _roots(tmp_path)
    old_flat = _flat_bytes(roots)
    backend = ScriptedRotationBackend(roots.state, _NEW, fail_probe=True)
    with pytest.raises(Exception):
        rotate_intent(
            INTENT,
            envs_root=roots.envs,
            target=TARGET,
            environment_command=("scripted-test-operator",),
            state_root=roots.state,
            env_root=roots.env,
            minimum_free_bytes=0,
            backend=backend,
            execution_lock_path=roots.state / ".test-execution.flock",
        )
    assert _flat_bytes(roots) == old_flat
    lock_dir = roots.envs / INTENT / "locks"
    assert not family_manifest_path(lock_dir, TARGET).is_file()
    assert not (lock_dir / "generations").exists()


def test_rotation_refuses_a_reused_solve_state_root(tmp_path: Path) -> None:
    """A pre-existing solve cache in the rotation state root refuses loudly.

    T3-T6 residual: ``--state-root`` is documented fresh but was never
    enforced, so a reused root could bless a stale name-complete solve into a
    fresh provenance.
    """

    roots = _roots(tmp_path)
    populate_solve_commit(roots.state, INTENT, _OLD)
    backend = ScriptedRotationBackend(roots.state, _NEW)
    with pytest.raises(RotationError, match="state root"):
        rotate_intent(
            INTENT,
            envs_root=roots.envs,
            target=TARGET,
            environment_command=("scripted-test-operator",),
            state_root=roots.state,
            env_root=roots.env,
            minimum_free_bytes=0,
            backend=backend,
            execution_lock_path=roots.state / ".test-execution.flock",
        )
    # Nothing was published.
    assert _current(roots).identity == _OLD_IDENTITY


# --- (3) rotate -> preflight end to end through main() (T5-11 RED, F) ----------


def test_rotate_then_preflight_releases_parks_end_to_end(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys,
    calibrated_checker: dict,
) -> None:
    """The production CLI wiring rotates, then the preflight releases parks.

    ``calibrated_checker`` stages the fixture calibration receipt: this e2e
    asserts the rotation/park path, not the W2 calibration gate, which keeps
    its own armed negative tests."""

    roots = _roots(tmp_path)
    _park_row(roots)

    preflight_argv = [
        "--records-root",
        str(roots.records),
        "--envs-root",
        str(roots.envs),
        "--target",
        TARGET,
    ]
    # Before the rotation the park row holds and the preflight refuses.
    assert preflight_main(preflight_argv) == 2
    capsys.readouterr()

    monkeypatch.setenv(
        "MENAGERIE_ENVIRONMENT_COMMAND",
        f"{sys.executable} -m menagerie.crawler.tests.scripted_environment_operator",
    )
    monkeypatch.setenv("MENAGERIE_SCRIPTED_OPERATOR_TAG", "new")
    monkeypatch.setenv("PYTHONPATH", str(_REPO_ROOT))
    campaign_state = tmp_path / "campaign-state"
    populate_solve_commit(campaign_state, INTENT, _OLD)
    code = rotate_main(
        [
            "--intent",
            INTENT,
            "--envs-root",
            str(roots.envs),
            "--target",
            TARGET,
            "--state-root",
            str(roots.state),
            "--env-root",
            str(roots.env),
            "--minimum-free-bytes",
            "0",
            "--campaign-state-root",
            str(campaign_state),
            "--execution-lock-path",
            str(roots.state / ".test-execution.flock"),
        ]
    )
    out = capsys.readouterr().out
    assert code == 0, out
    result = json.loads(out.strip().splitlines()[-1])
    assert result["rotated"] is True
    assert result["old_identity"] == _OLD_IDENTITY
    assert result["new_identity"] == _NEW_IDENTITY

    disposition = _current(roots)
    assert disposition.status == "locked"
    assert disposition.identity == _NEW_IDENTITY
    # The campaign cache serves exactly the committed bytes.
    assert (
        campaign_state / "solves" / INTENT / TARGET / "explicit.lock"
    ).read_bytes() == _NEW.lock_bytes

    # After the rotation the park releases and every family assert passes.
    assert preflight_main(preflight_argv) == 0
    parks = evaluate_rebuild_parks(records_root=roots.records, envs_root=roots.envs, target=TARGET)
    assert parks["released"] == 1 and parks["still_held"] == 0


# --- (4) preflight family validation: mixed families refuse loudly (T3-T7) -----


def test_preflight_refuses_a_mixed_family_loudly(tmp_path: Path, capsys) -> None:
    """A member byte change without its provenance is a MIXED family, not a
    rotation: the preflight refuses with a typed token instead of releasing
    parks against an unproven family."""

    roots = _roots(tmp_path)
    _park_row(roots)
    lock_path = roots.envs / INTENT / "locks" / f"{TARGET}.lock"
    lock_path.write_bytes(lock_path.read_bytes() + b"# hand edit\n")

    integrity = evaluate_family_integrity(envs_root=roots.envs, target=TARGET, allow_lockless=())
    assert integrity["ok"] is False
    assert any(row["reason"] == MIXED_FAMILY for row in integrity["refused"])

    code = preflight_main(
        [
            "--records-root",
            str(roots.records),
            "--envs-root",
            str(roots.envs),
            "--target",
            TARGET,
        ]
    )
    captured = capsys.readouterr()
    assert code == 2
    assert MIXED_FAMILY in captured.err


@pytest.mark.parametrize(
    "member",
    [f"{TARGET}.resolved.json", f"{TARGET}.probes.json"],
)
def test_every_provenance_bound_member_is_validated(tmp_path: Path, member: str) -> None:
    """Each provenance-bound member mismatch reads as a mixed family."""

    roots = _roots(tmp_path)
    path = roots.envs / INTENT / "locks" / member
    if member.endswith(".resolved.json"):
        # Keep the declared export hash consistent so ONLY the provenance
        # binding can catch the swap (the T3-T7 window: identity changed,
        # nothing validated the family).
        mutated = _NEW.export_bytes
        path.write_bytes(mutated)
        (roots.envs / INTENT / "locks" / f"{TARGET}.resolved.sha256").write_text(
            f"{hash_bytes(mutated)}\n", encoding="utf-8"
        )
    else:
        # Stays parseable against the declared probes; ONLY the provenance
        # binding can catch the byte change.
        path.write_bytes(path.read_bytes().replace(b'"ok"', b'"tampered"', 1))
    integrity = evaluate_family_integrity(envs_root=roots.envs, target=TARGET, allow_lockless=())
    assert integrity["ok"] is False
    assert any(row["reason"] == MIXED_FAMILY for row in integrity["refused"])


def test_a_family_without_provenance_is_refused(tmp_path: Path) -> None:
    """A locked family whose provenance member is missing proves nothing."""

    roots = _roots(tmp_path)
    (roots.envs / INTENT / "locks" / f"{TARGET}.provenance.json").unlink()
    integrity = evaluate_family_integrity(envs_root=roots.envs, target=TARGET, allow_lockless=())
    assert integrity["ok"] is False
    assert any(row["reason"] == MIXED_FAMILY for row in integrity["refused"])


def test_allow_lockless_intents_are_exempt_from_integrity(tmp_path: Path) -> None:
    """Deliberately lockless intents never enter the family validation."""

    roots = _roots(tmp_path)
    integrity = evaluate_family_integrity(
        envs_root=roots.envs, target=TARGET, allow_lockless=(INTENT,)
    )
    assert integrity["ok"] is True


# --- (5) spec and probe currency: post-rotation edits read stale (T3-T6) --------


def test_a_post_rotation_spec_edit_reads_stale_not_current(tmp_path: Path, capsys) -> None:
    """Editing environment.yml after the rotation refuses as STALE_SPEC.

    The rotation-lock-provenance.v1 lesson: ``spec_sha256`` existed from the
    first rotation but was compared by NOTHING at runtime, so bound edits,
    removals, and channel edits all read current.
    """

    roots = _roots(tmp_path)
    spec = roots.envs / INTENT / "environment.yml"
    spec.write_text(
        json.dumps(
            {
                "name": INTENT,
                "channels": ["conda-forge", "extra-channel"],
                "dependencies": ["python"],
            }
        ),
        encoding="utf-8",
    )
    integrity = evaluate_family_integrity(envs_root=roots.envs, target=TARGET, allow_lockless=())
    assert integrity["ok"] is False
    assert any(row["reason"] == STALE_SPEC for row in integrity["refused"])

    code = preflight_main(
        [
            "--records-root",
            str(roots.records),
            "--envs-root",
            str(roots.envs),
            "--target",
            TARGET,
        ]
    )
    captured = capsys.readouterr()
    assert code == 2
    assert STALE_SPEC in captured.err


def test_a_post_rotation_probe_declaration_edit_reads_stale(tmp_path: Path) -> None:
    """A probes.yml edit for THIS intent refuses; sibling edits never do."""

    roots = _roots(tmp_path)
    probes_path = roots.envs / "probes.yml"
    declaration = json.loads(probes_path.read_text(encoding="utf-8"))
    declaration["intents"][INTENT]["imports"] = ["a", "b", "c", "d"]
    probes_path.write_text(json.dumps(declaration), encoding="utf-8")
    integrity = evaluate_family_integrity(envs_root=roots.envs, target=TARGET, allow_lockless=())
    assert integrity["ok"] is False
    assert any(row["reason"] == STALE_PROBE_RECEIPT for row in integrity["refused"])


def test_a_sibling_probe_contract_edit_never_reads_stale(tmp_path: Path) -> None:
    """Whole-file probe-contract hashing was structurally always-stale (8/9);
    the per-intent comparison must not inherit that false positive."""

    roots = _roots(tmp_path)
    probes_path = roots.envs / "probes.yml"
    declaration = json.loads(probes_path.read_text(encoding="utf-8"))
    declaration["intents"]["sibling"] = {"imports": ["x", "y", "z"], "export_checks": []}
    registry = json.loads((roots.envs / "registry.yml").read_text(encoding="utf-8"))
    registry["intents"]["sibling"] = {
        "phase": "pytorch",
        "framework": "pytorch",
        "description": "sibling",
        "split_guidance": "sibling",
    }
    probes_path.write_text(json.dumps(declaration), encoding="utf-8")
    (roots.envs / "registry.yml").write_text(json.dumps(registry), encoding="utf-8")
    sibling_dir = roots.envs / "sibling"
    sibling_dir.mkdir()
    (sibling_dir / "environment.yml").write_text(
        json.dumps({"name": "sibling", "channels": ["conda-forge"], "dependencies": ["python"]}),
        encoding="utf-8",
    )
    integrity = evaluate_family_integrity(
        envs_root=roots.envs, target=TARGET, allow_lockless=("sibling",)
    )
    assert integrity["ok"] is True, integrity["refused"]


# --- (6) the generation authority and its fail-closed loader -------------------


def test_crash_mid_materialization_serves_the_generation_authority(
    tmp_path: Path,
) -> None:
    """After the manifest switch the generation directory is the family.

    A crash between flat member writes leaves the manifest committed; the
    loader serves the complete proven generation bytes -- never the mixed flat
    view -- and the preflight refuses until the materialization is completed.
    """

    roots = _roots(tmp_path)
    lock_dir = roots.envs / INTENT / "locks"
    members = family_members(_NEW, rotation_provenance(roots.envs, INTENT, _NEW))

    class Boom(RuntimeError):
        pass

    calls = {"n": 0}

    def crash_between_flat_writes(name: str) -> None:
        calls["n"] += 1
        if calls["n"] == 2:
            raise Boom("simulated crash between flat member writes")

    with pytest.raises(Boom):
        publish_rotation_family(lock_dir, TARGET, members, observer=crash_between_flat_writes)

    disposition = _current(roots)
    assert disposition.status == "locked"
    assert disposition.identity == _NEW_IDENTITY
    assert disposition.intent.lock.lock_bytes == _NEW.lock_bytes
    assert disposition.intent.lock.materialization_pending is True

    integrity = evaluate_family_integrity(envs_root=roots.envs, target=TARGET, allow_lockless=())
    assert integrity["ok"] is False
    assert any(row["reason"] == MATERIALIZATION_PENDING for row in integrity["refused"])

    report = complete_family_publication(lock_dir, TARGET)
    assert report["completed"] is True
    assert _flat_bytes(roots)[f"{TARGET}.lock"] == _NEW.lock_bytes
    assert not publication_journal_path(lock_dir, TARGET).is_file()
    integrity = evaluate_family_integrity(envs_root=roots.envs, target=TARGET, allow_lockless=())
    assert integrity["ok"] is True, integrity["refused"]


def test_newer_proven_flat_family_outranks_a_stale_host_manifest(tmp_path: Path) -> None:
    """A pull from another rotation host can never be rolled back locally."""

    roots = _roots(tmp_path)
    lock_dir = roots.envs / INTENT / "locks"
    old_members = family_members(_OLD, rotation_provenance(roots.envs, INTENT, _OLD))
    publish_rotation_family(lock_dir, TARGET, old_members)
    new_members = family_members(_NEW, rotation_provenance(roots.envs, INTENT, _NEW))
    for name, content in new_members.items():
        (lock_dir / name).write_bytes(content)

    disposition = _current(roots)
    assert disposition.status == "locked"
    assert disposition.identity == _NEW_IDENTITY
    assert disposition.intent.lock.family_authority == "flat-newer-than-manifest"
    report = complete_family_publication(lock_dir, TARGET)
    assert report["stale_manifest_discarded"] is True
    assert _flat_bytes(roots)[f"{TARGET}.lock"] == _NEW.lock_bytes
    assert not family_manifest_path(lock_dir, TARGET).exists()


def test_an_invalid_or_torn_family_manifest_fails_closed(tmp_path: Path) -> None:
    """A manifest that cannot prove its generation refuses as publishing."""

    roots = _roots(tmp_path)
    lock_dir = roots.envs / INTENT / "locks"
    manifest = family_manifest_path(lock_dir, TARGET)

    # Unparseable manifest bytes fail closed.
    manifest.write_bytes(b"{torn")
    disposition = _current(roots)
    assert disposition.status == "publishing"
    assert disposition.intent.generation is None

    # A manifest naming a generation whose member digest does not verify
    # fails closed too -- the generation directory is IMMUTABLE evidence.
    members = family_members(_NEW, rotation_provenance(roots.envs, INTENT, _NEW))
    publish_rotation_family(lock_dir, TARGET, members)
    assert _current(roots).status == "locked"
    generation_dir = next(path for path in (lock_dir / "generations").iterdir() if path.is_dir())
    (generation_dir / f"{TARGET}.lock").write_bytes(b"corrupted\n")
    assert _current(roots).status == "publishing"


def test_a_deleted_manifest_with_mixed_flat_members_is_still_caught(
    tmp_path: Path,
) -> None:
    """Losing the manifest cannot resurrect a mixed flat family as proven.

    The provenance binding is the independent detector: flat members from two
    generations never satisfy one provenance record.
    """

    roots = _roots(tmp_path)
    lock_dir = roots.envs / INTENT / "locks"
    members = family_members(_NEW, rotation_provenance(roots.envs, INTENT, _NEW))

    class Boom(RuntimeError):
        pass

    calls = {"n": 0}

    def crash(name: str) -> None:
        calls["n"] += 1
        if calls["n"] == 2:
            raise Boom("crash")

    with pytest.raises(Boom):
        publish_rotation_family(lock_dir, TARGET, members, observer=crash)
    family_manifest_path(lock_dir, TARGET).unlink()
    shutil.rmtree(lock_dir / "generations")
    publication_journal_path(lock_dir, TARGET).unlink(missing_ok=True)

    integrity = evaluate_family_integrity(envs_root=roots.envs, target=TARGET, allow_lockless=())
    assert integrity["ok"] is False
    assert any(row["reason"] == MIXED_FAMILY for row in integrity["refused"])


def test_family_runtime_state_is_invisible_to_git() -> None:
    """Generations, manifests, and journals never become tracked artifacts.

    The anti-fabrication gate scans every tracked solve output; runtime
    publication state must stay outside its universe by ignore rule, not by
    hope.
    """

    lock_dir = Path("menagerie/crawler/envs") / INTENT / "locks"
    for candidate in (
        lock_dir / "generations" / "gen" / f"{TARGET}.lock",
        lock_dir / f"{TARGET}.family.json",
        lock_dir / f".{TARGET}.publication-journal.json",
    ):
        completed = subprocess.run(
            ["git", "check-ignore", "-q", str(candidate)],
            cwd=_REPO_ROOT,
            capture_output=True,
        )
        assert completed.returncode == 0, f"{candidate} is not git-ignored"


def test_republication_refuses_a_stale_unified_lock(tmp_path: Path) -> None:
    """A torn rotation solve cache cannot republish a stale conda-lock.yml.

    The operator's solve-commit binds the unified lock into its manifest;
    a unified lock that fails that binding (a crash between solve-commit
    member writes) refuses instead of silently shipping another solve's
    bytes into the campaign cache.
    """

    from menagerie.crawler.tools.rotate_setup_artifacts import (
        republish_campaign_solve_cache,
    )

    rotation_state = tmp_path / "rotation-state"
    populate_solve_commit(rotation_state, INTENT, _NEW)
    solve_root = rotation_state / "solves" / INTENT / TARGET
    (solve_root / "conda-lock.yml").write_bytes(b"# stale unified lock\n")

    campaign_state = tmp_path / "campaign-state"
    with pytest.raises(RotationError, match="unified"):
        republish_campaign_solve_cache(
            campaign_state,
            rotation_state_root=rotation_state,
            intent=INTENT,
            target=TARGET,
            lock_bytes=_NEW.lock_bytes,
            export_bytes=_NEW.export_bytes,
        )


# --- (7) the real committed registry passes its own validation ------------------


def test_the_committed_registry_families_validate_clean() -> None:
    """Every real committed family passes integrity: the gate arms tightly.

    A validator the real repo cannot satisfy would be disabled on day one;
    this pins that the 11 committed rotation families -- v1 provenance, current
    specs, current probe receipts -- all read clean. ``mmlab`` and ``oddballs``
    graduated to locked families in the 2026-08-19 rotations, so they are
    validated here rather than exempted as lockless.
    """

    from menagerie.crawler.envs import DEFAULT_ENVS_ROOT

    integrity = evaluate_family_integrity(
        envs_root=DEFAULT_ENVS_ROOT,
        target=TARGET,
        allow_lockless=("tf-keras-arm64", "jax-flax-arm64"),
    )
    assert integrity["ok"] is True, integrity["refused"]
